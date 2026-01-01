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
    try:
        ensemble.train_on_composition(feat, n_samples=500, epochs=50)
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
            
            # Waveform crossfade score
            crossfade_score = crossfade_scorer.score(
                candidate._loop_start_frame_idx,
                candidate._loop_end_frame_idx
            )
            
            # Combine scores (ensemble is primary, crossfade is validation)
            candidate.score = ensemble_score * 0.75 + crossfade_score * 0.25
            
        except Exception as e:
            logging.debug(f"Failed to score candidate: {e}")
            candidate.score = 0.0
    
    # Filter low scores
    threshold = max(0.25, np.percentile([c.score for c in candidates], 15))
    scored = [c for c in candidates if c.score >= threshold]
    
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
    for click-free transitions.
    """
    from pymusiclooper.analysis.features import nearest_zero_crossing
    
    for pair in scored:
        # Convert frames to samples
        start_samples = mlaudio.frames_to_samples(
            mlaudio.apply_trim_offset(pair._loop_start_frame_idx)
        )
        end_samples = mlaudio.frames_to_samples(
            mlaudio.apply_trim_offset(pair._loop_end_frame_idx)
        )
        
        # Align to zero-crossings for click-free transitions
        pair.loop_start = nearest_zero_crossing(
            mlaudio.playback_audio, mlaudio.rate, start_samples
        )
        pair.loop_end = nearest_zero_crossing(
            mlaudio.playback_audio, mlaudio.rate, end_samples
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

