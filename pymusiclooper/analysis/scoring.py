"""
Candidate Scoring Module.

Combines all scoring mechanisms:
- Expert ensemble scoring
- Waveform crossfade scoring
- Rhythm gating
- Final optimization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from pymusiclooper.audio import MLAudio
    from pymusiclooper.analysis.features import Features
    from pymusiclooper.analysis.candidates import LoopPair

from pymusiclooper.analysis.experts.base import TransitionContext
from pymusiclooper.analysis.experts.ensemble import ExpertEnsemble, WaveformCrossfadeScorer


def score_candidates(
    mlaudio: MLAudio,
    feat: Features,
    candidates: list[LoopPair],
    disable_pruning: bool = False,
) -> list[LoopPair]:
    """
    Score all candidates using the expert ensemble.
    
    This is the main scoring function that combines:
    1. Expert ensemble (harmonic, rhythm, timbre, etc.)
    2. Waveform crossfade quality
    3. Rhythm gating (hard filter for bad rhythm)
    4. Pruning of low-quality candidates
    """
    if not candidates:
        return []
    
    # Prune if too many candidates
    if len(candidates) >= 80 and not disable_pruning:
        candidates = _prune_candidates(candidates)
    
    n = len(candidates)
    logging.info(f"Scoring {n} candidates...")
    
    # Initialize scorers
    ensemble = ExpertEnsemble()
    crossfade_scorer = WaveformCrossfadeScorer(feat.audio, feat.sr, feat.hop)
    
    # Train ensemble on this composition for adaptive weights
    # Use more samples and epochs for better adaptation
    try:
        n_samples = min(1000, len(feat.beats) * 10)  # Adaptive sample count
        epochs = 100  # More epochs for better convergence
        ensemble.train_on_composition(feat, n_samples=n_samples, epochs=epochs)
    except Exception as e:
        logging.debug(f"Ensemble training skipped: {e}")
    
    # Score each candidate
    for candidate in candidates:
        try:
            # Create transition context
            ctx = TransitionContext.from_features(
                feat, 
                candidate._loop_start_frame_idx, 
                candidate._loop_end_frame_idx
            )
            
            # Expert ensemble score
            ensemble_score = ensemble.score(ctx)
            
            from pymusiclooper.analysis.constants import (
                TIMBRE_PENALTY_VERY_BAD,
                TIMBRE_PENALTY_BAD,
                TIMBRE_PENALTY_MULT_VERY_BAD,
                TIMBRE_PENALTY_MULT_BAD,
                ENSEMBLE_WEIGHT,
                CROSSFADE_WEIGHT,
                BEAT_PHASE_ACCEPTABLE,
                compute_adaptive_timbre_threshold
            )
            from pymusiclooper.analysis.experts.base import circular_distance
            
            # HARD RHYTHM GATING - reject candidates with poor rhythm alignment
            # This is CRITICAL for preventing misrhythmic transitions
            rhythm_expert = ensemble.experts[1]  # RhythmExpert
            rhythm_score = rhythm_expert.score(ctx)
            
            # Get beat phase difference for strict gating
            phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
            
            # Check if we have drums (high transients)
            avg_transient = (ctx.transient_start + ctx.transient_end) / 2
            has_drums = avg_transient > 0.4
            
            # ULTRA-STRICT rhythm gate for drums
            if has_drums:
                # For drums, reject if phase difference is too large
                if phase_diff > BEAT_PHASE_ACCEPTABLE * 0.6:
                    candidate.score = 0.0  # HARD REJECT
                    continue
                # Also reject if rhythm score is too low
                if rhythm_score < 0.5:
                    candidate.score = 0.0  # HARD REJECT
                    continue
            else:
                # For non-drums, still strict but slightly more lenient
                if phase_diff > BEAT_PHASE_ACCEPTABLE * 0.9:
                    candidate.score = 0.0  # HARD REJECT
                    continue
                if rhythm_score < 0.35:
                    candidate.score = 0.0  # HARD REJECT
                    continue
            
            # Get timbre score separately for soft filtering
            timbre_score = ensemble.experts[2].score(ctx)  # TimbreExpert
            
            # HARD VOCAL/ENSEMBLE GATING - reject candidates where vocals/instruments appear/disappear
            vocal_expert = ensemble.experts[16]  # VocalExpert
            vocal_score = vocal_expert.score(ctx)
            
            # Hard reject if vocal score is very low (vocals appear/disappear)
            if vocal_score < 0.1:
                candidate.score = 0.0  # HARD REJECT
                continue
            
            # Check ensemble instruments
            ensemble_expert = ensemble.experts[17]  # EnsembleInstrumentsExpert
            ensemble_score_val = ensemble_expert.score(ctx)
            
            # Hard reject if ensemble score is very low (instruments appear/disappear)
            if ensemble_score_val < 0.1:
                candidate.score = 0.0  # HARD REJECT
                continue
            
            # Waveform crossfade score
            crossfade_score = crossfade_scorer.score(
                candidate._loop_start_frame_idx,
                candidate._loop_end_frame_idx
            )
            
            # Combine scores (ensemble is primary, crossfade is validation)
            base_score = ensemble_score * ENSEMBLE_WEIGHT + crossfade_score * CROSSFADE_WEIGHT
            
            # Apply vocal/ensemble penalty for borderline cases
            if vocal_score < 0.3:
                # Poor vocal match - significant penalty
                vocal_penalty = 1.0 - vocal_score
                base_score *= (1.0 - vocal_penalty * 0.4)  # Up to 28% penalty
            
            # Apply rhythm penalty for borderline cases (not rejected but not perfect)
            if has_drums:
                if rhythm_score < 0.7:
                    # Penalize poor rhythm even if not rejected
                    rhythm_penalty = 1.0 - (rhythm_score - 0.5) * 2  # 0.0 to 0.4 penalty
                    base_score *= (1.0 - rhythm_penalty * 0.3)  # Up to 12% penalty
            else:
                if rhythm_score < 0.6:
                    rhythm_penalty = 1.0 - (rhythm_score - 0.35) * 4  # 0.0 to 0.6 penalty
                    base_score *= (1.0 - rhythm_penalty * 0.2)  # Up to 12% penalty
            
            # Soft timbre penalty - only for very bad matches
            # This catches cases like guitar at start but no guitar at end
            if timbre_score < TIMBRE_PENALTY_VERY_BAD:
                # Very bad timbre match - moderate penalty
                candidate.score = base_score * TIMBRE_PENALTY_MULT_VERY_BAD
            elif timbre_score < TIMBRE_PENALTY_BAD:
                # Poor timbre match - light penalty
                candidate.score = base_score * TIMBRE_PENALTY_MULT_BAD
            else:
                # Good timbre match - normal scoring
                candidate.score = base_score
            
        except Exception as e:
            logging.debug(f"Failed to score candidate: {e}")
            candidate.score = 0.0
    
    # Filter low scores - vectorized
    scores_array = np.array([c.score for c in candidates], dtype=np.float32)
    threshold = max(0.25, np.percentile(scores_array, 15))
    scored = [c for c, s in zip(candidates, scores_array) if s >= threshold]
    
    if not scored:
        # Keep at least some candidates
        candidates.sort(key=lambda x: x.score, reverse=True)
        scored = candidates[:max(5, len(candidates) // 10)]
    
    logging.info(f"After scoring: {len(scored)} candidates remain")
    
    return scored


def finalize_with_alignment(
    mlaudio: MLAudio,
    feat: Features,
    scored: list[LoopPair],
) -> None:
    """
    Finalize loop points with sample-accurate alignment.
    
    Adjusts frame positions to samples and aligns to zero crossings
    for click-free transitions, while preserving rhythm alignment.
    
    Strategy:
    1. If point is far from beat (phase > 0.2), try to snap to nearest beat first
    2. Then align to zero-crossing with tight window to preserve rhythm
    """
    from pymusiclooper.analysis.features import nearest_zero_crossing
    import numpy as np
    
    for pair in scored:
        # Convert frames to samples
        start_frame = mlaudio.apply_trim_offset(pair._loop_start_frame_idx)
        end_frame = mlaudio.apply_trim_offset(pair._loop_end_frame_idx)
        
        start_samples = mlaudio.frames_to_samples(start_frame)
        end_samples = mlaudio.frames_to_samples(end_frame)
        
        from pymusiclooper.analysis.constants import BEAT_PHASE_ACCEPTABLE
        
        # Get beat phases and transient strengths to preserve rhythm alignment
        start_phase = float(feat.beat_phase[min(start_frame, len(feat.beat_phase) - 1)])
        end_phase = float(feat.beat_phase[min(end_frame, len(feat.beat_phase) - 1)])
        start_transient = float(feat.transient_strength[min(start_frame, len(feat.transient_strength) - 1)])
        end_transient = float(feat.transient_strength[min(end_frame, len(feat.transient_strength) - 1)])
        
        # ULTRA-PRECISE beat snapping - CRITICAL for rhythm preservation
        # For drums (high transients), be even more aggressive with beat snapping
        if len(feat.beats) > 0:
            beats_samples = np.array([mlaudio.frames_to_samples(b) for b in feat.beats])
            
            # Determine snap tolerance based on transient strength
            if start_transient > 0.4 or end_transient > 0.4:
                # Drums - very tight tolerance (20ms)
                snap_tolerance_ms = 20
            else:
                # Non-drums - slightly more lenient (30ms)
                snap_tolerance_ms = 30
            
            snap_tolerance_samples = int(mlaudio.rate * snap_tolerance_ms / 1000)
            
            # Snap start to nearest beat if phase is off
            if start_phase > BEAT_PHASE_ACCEPTABLE * 0.8 and start_phase < (1.0 - BEAT_PHASE_ACCEPTABLE * 0.8):
                nearest_beat_idx = np.argmin(np.abs(beats_samples - start_samples))
                nearest_beat = beats_samples[nearest_beat_idx]
                # Snap if close enough
                if abs(nearest_beat - start_samples) < snap_tolerance_samples:
                    start_samples = nearest_beat
                    start_phase = 0.0  # Now on beat
            
            # Snap end to nearest beat if phase is off
            if end_phase > BEAT_PHASE_ACCEPTABLE * 0.8 and end_phase < (1.0 - BEAT_PHASE_ACCEPTABLE * 0.8):
                nearest_beat_idx = np.argmin(np.abs(beats_samples - end_samples))
                nearest_beat = beats_samples[nearest_beat_idx]
                if abs(nearest_beat - end_samples) < snap_tolerance_samples:
                    end_samples = nearest_beat
                    end_phase = 0.0
        
        # Align to zero-crossings while preserving beat phase and transient timing
        # CRITICAL: For drums (high transients), preserve precise beat timing
        pair.loop_start = nearest_zero_crossing(
            mlaudio.playback_audio, 
            mlaudio.rate, 
            start_samples, 
            beat_phase=start_phase,
            transient_strength=start_transient
        )
        pair.loop_end = nearest_zero_crossing(
            mlaudio.playback_audio, 
            mlaudio.rate, 
            end_samples, 
            beat_phase=end_phase,
            transient_strength=end_transient
        )


def _prune_candidates(candidates: list[LoopPair], max_keep: int = 100) -> list[LoopPair]:
    """Prune candidates based on initial heuristics."""
    if len(candidates) <= max_keep:
        return candidates
    
    # Sort by combined heuristic
    candidates.sort(key=lambda x: (x.note_distance * 0.7 + x.loudness_difference * 0.3))
    
    n_keep = min(len(candidates), max(max_keep, len(candidates) // 4))
    return candidates[:n_keep]


def optimize_selection(scored: list[LoopPair], feat: Features) -> None:
    """
    Apply small bonuses for ideal musical structures.
    
    Encourages loops that align with common musical phrase lengths
    while keeping scores in [0, 1].
    """
    if len(scored) <= 3:
        return
    
    bar_len = feat.bar_length
    beat_len = feat.beat_length
    
    for p in scored:
        dur = p._loop_end_frame_idx - p._loop_start_frame_idx
        
        # Small bonus for ideal bar lengths (max 0.05)
        bar_bonus = 0.0
        if bar_len > 0:
            bars = dur / bar_len
            for target in [4, 8, 16, 32]:
                if abs(bars - target) < 0.5:
                    bar_bonus = (1.0 - abs(bars - target) * 2) * 0.05
                    break
        
        # Small bonus for beat alignment (max 0.03)
        beat_bonus = 0.0
        if beat_len > 0:
            start_on = p._loop_start_frame_idx % beat_len < beat_len * 0.15
            end_on = p._loop_end_frame_idx % beat_len < beat_len * 0.15
            beat_bonus = (float(start_on) * 0.5 + float(end_on) * 0.5) * 0.03
        
        # Apply bonuses, keep score in [0, 1]
        p.score = min(1.0, p.score + bar_bonus + beat_bonus)


# ============================================================================
# LEGACY SCORING FUNCTIONS (for compatibility)
# ============================================================================

@njit(cache=True, parallel=True, fastmath=True)
def _score_phase_vec(
    beat_phase: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    n_frames: int
) -> np.ndarray:
    """Vectorized beat phase scoring."""
    n = len(starts)
    n_f = len(beat_phase)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        p1 = beat_phase[b1]
        p2 = beat_phase[b2]
        
        diff = abs(p1 - p2)
        if diff > 0.5:
            diff = 1.0 - diff
        
        if diff > 0.15:
            scores[i] = 0.1
        elif diff > 0.08:
            scores[i] = 0.4
        elif diff > 0.04:
            scores[i] = 0.7
        else:
            scores[i] = 1.0 - diff * 8
        
        on_beat_1 = p1 < 0.06 or p1 > 0.94
        on_beat_2 = p2 < 0.06 or p2 > 0.94
        if on_beat_1 and on_beat_2:
            scores[i] = min(1.0, scores[i] + 0.15)
        
        scores[i] = max(0.0, min(1.0, scores[i]))
    
    return scores


@njit(cache=True, parallel=True, fastmath=True)
def _score_transient_vec(
    transient: np.ndarray,
    onset_peaks: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    n_frames: int
) -> np.ndarray:
    """Vectorized transient avoidance scoring."""
    n = len(starts)
    n_f = len(transient)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        t1 = transient[b1]
        t2 = transient[b2]
        
        if t1 > 0.7 or t2 > 0.7:
            scores[i] = 0.1
            continue
        if t1 > 0.5 or t2 > 0.5:
            scores[i] = 0.3
            continue
        
        transient_score = 1.0 - (t1 + t2)
        
        o1 = onset_peaks[b1]
        o2 = onset_peaks[b2]
        if o1 > 0.5 or o2 > 0.5:
            scores[i] = 0.2
            continue
        
        scores[i] = max(0.0, min(1.0, transient_score))
    
    return scores

