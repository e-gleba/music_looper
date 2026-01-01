"""
Segment Loop Finder - Find the best loop for a user-selected segment.

This is the core algorithm for when a user selects a favorite moment
and wants to find the best way to loop it seamlessly.

Algorithm:
1. Extract features from the segment
2. Find similar sections within the segment using self-similarity
3. Score candidates using expert ensemble
4. Optimize boundaries for seamless transitions
5. Return ranked loop points that respect user's selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.ndimage import uniform_filter1d

if TYPE_CHECKING:
    from pymusiclooper.audio import MLAudio
    from pymusiclooper.analysis.features import Features

from pymusiclooper.analysis.candidates import LoopPair
from pymusiclooper.analysis.experts.base import TransitionContext
from pymusiclooper.analysis.experts.ensemble import ExpertEnsemble, WaveformCrossfadeScorer


@dataclass
class SegmentLoopResult:
    """Result of segment loop analysis."""
    loop_start: int          # Sample position
    loop_end: int            # Sample position
    score: float             # Overall quality score
    boundary_type: str       # 'exact', 'optimized', 'extended'
    explanation: str         # Human-readable explanation
    
    # For analysis
    start_frame: int
    end_frame: int
    transition_quality: float
    rhythm_alignment: float
    harmonic_match: float


def find_segment_loop_points(
    mlaudio: MLAudio,
    feat: Features,
    segment_start_sec: float,
    segment_end_sec: float,
    search_mode: str = 'internal',  # 'internal', 'boundary', 'extended'
    n_results: int = 10,
) -> list[SegmentLoopResult]:
    """
    Find optimal loop points for a user-selected audio segment.
    
    The algorithm tries to find the best way to loop the selected moment,
    with different strategies:
    
    - 'internal': Find loops WITHIN the segment (repeating patterns inside)
    - 'boundary': Optimize the exact start/end of the segment for seamless loop
    - 'extended': Allow slight extension beyond segment to find better loops
    
    Args:
        mlaudio: Audio object
        feat: Extracted features
        segment_start_sec: User's selected start time in seconds
        segment_end_sec: User's selected end time in seconds
        search_mode: Search strategy ('internal', 'boundary', 'extended')
        n_results: Number of results to return
        
    Returns:
        List of SegmentLoopResult sorted by score (best first)
    """
    logging.info(f"Finding segment loops: {segment_start_sec:.2f}s - {segment_end_sec:.2f}s ({search_mode})")
    
    # Convert to frames
    start_frame = mlaudio.seconds_to_frames(segment_start_sec)
    end_frame = mlaudio.seconds_to_frames(segment_end_sec)
    
    # Clamp to valid range
    start_frame = max(0, min(start_frame, feat.n_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, feat.n_frames))
    
    segment_length = end_frame - start_frame
    
    if segment_length < feat.beat_length * 2:
        logging.warning("Segment too short for reliable loop detection")
    
    # Initialize ensemble
    ensemble = ExpertEnsemble()
    
    # Initialize waveform scorer
    crossfade_scorer = WaveformCrossfadeScorer(feat.audio, feat.sr, feat.hop)
    
    # Strategy dispatch
    if search_mode == 'internal':
        candidates = _find_internal_loops(feat, start_frame, end_frame)
    elif search_mode == 'boundary':
        candidates = _optimize_boundaries(feat, start_frame, end_frame)
    else:  # extended
        candidates = _find_extended_loops(feat, start_frame, end_frame)
    
    if not candidates:
        # Fallback: just optimize the exact boundaries
        candidates = _optimize_boundaries(feat, start_frame, end_frame)
    
    # Score all candidates
    results = []
    for loop_start_frame, loop_end_frame, boundary_type in candidates:
        try:
            # Get transition context
            ctx = TransitionContext.from_features(feat, loop_start_frame, loop_end_frame)
            
            # Expert ensemble score
            ensemble_score = ensemble.score(ctx)
            
            # Waveform crossfade score
            crossfade_score = crossfade_scorer.score(loop_start_frame, loop_end_frame)
            
            # Combined score
            final_score = ensemble_score * 0.7 + crossfade_score * 0.3
            
            # Get detailed component scores for explanation
            rhythm_score = ensemble.experts[1].score(ctx)  # RhythmExpert
            harmonic_score = ensemble.experts[0].score(ctx)  # HarmonicExpert
            
            # Convert to samples with zero-crossing alignment
            from pymusiclooper.analysis.features import nearest_zero_crossing
            loop_start_samples = mlaudio.frames_to_samples(
                mlaudio.apply_trim_offset(loop_start_frame)
            )
            loop_end_samples = mlaudio.frames_to_samples(
                mlaudio.apply_trim_offset(loop_end_frame)
            )
            
            loop_start_samples = nearest_zero_crossing(
                mlaudio.playback_audio, mlaudio.rate, loop_start_samples
            )
            loop_end_samples = nearest_zero_crossing(
                mlaudio.playback_audio, mlaudio.rate, loop_end_samples
            )
            
            # Generate explanation
            explanation = _generate_explanation(
                ctx, ensemble_score, crossfade_score, 
                rhythm_score, harmonic_score, boundary_type
            )
            
            results.append(SegmentLoopResult(
                loop_start=loop_start_samples,
                loop_end=loop_end_samples,
                score=final_score,
                boundary_type=boundary_type,
                explanation=explanation,
                start_frame=loop_start_frame,
                end_frame=loop_end_frame,
                transition_quality=crossfade_score,
                rhythm_alignment=rhythm_score,
                harmonic_match=harmonic_score,
            ))
            
        except Exception as e:
            logging.debug(f"Failed to score candidate: {e}")
            continue
    
    # Sort by score (best first)
    results.sort(key=lambda x: x.score, reverse=True)
    
    # Deduplicate (remove results too close to each other)
    deduped = _deduplicate_results(results, min_distance_frames=feat.beat_length)
    
    logging.info(f"Found {len(deduped)} segment loops")
    
    return deduped[:n_results]


def _find_internal_loops(
    feat: Features,
    start_frame: int,
    end_frame: int,
) -> list[tuple[int, int, str]]:
    """
    Find repeating patterns WITHIN the segment.
    
    Uses autocorrelation and self-similarity to find
    periodic structures that can be looped.
    """
    candidates = []
    segment_length = end_frame - start_frame
    
    # 1. Autocorrelation-based periodicity detection
    segment_chroma = feat.chroma_cens[:, start_frame:end_frame]
    segment_rms = feat.rms[start_frame:end_frame]
    
    # Compute autocorrelation of the segment
    autocorr = _compute_chroma_autocorr(segment_chroma)
    
    # Find peaks in autocorrelation (periodic lengths)
    min_period = max(feat.beat_length * 2, segment_length // 8)
    max_period = segment_length // 2
    
    if min_period < max_period:
        peak_indices, _ = find_peaks(
            autocorr[min_period:max_period],
            distance=feat.beat_length,
            height=0.5
        )
        
        for peak_idx in peak_indices[:5]:  # Top 5 periods
            period = min_period + peak_idx
            
            # Generate candidates at this period
            for offset in range(0, segment_length - period, feat.beat_length):
                loop_start = start_frame + offset
                loop_end = start_frame + offset + period
                
                if loop_end <= end_frame:
                    candidates.append((loop_start, loop_end, 'internal'))
    
    # 2. Self-similarity based detection within segment
    ssm_segment = feat.ssm_chroma[start_frame:end_frame, start_frame:end_frame] \
        if start_frame < len(feat.ssm_chroma) and end_frame <= len(feat.ssm_chroma) \
        else np.eye(segment_length)
    
    # Find diagonal stripes (repeating sections)
    for offset in range(min_period, max_period):
        if offset >= ssm_segment.shape[0]:
            break
            
        diag = np.diag(ssm_segment, k=offset)
        if len(diag) < 2:
            continue
            
        # Find high-similarity regions
        threshold = 0.7
        high_sim = diag > threshold
        
        if np.any(high_sim):
            # Find start of first high-similarity region
            first_high = np.argmax(high_sim)
            loop_start = start_frame + first_high
            loop_end = loop_start + offset
            
            if loop_end <= end_frame:
                candidates.append((loop_start, loop_end, 'internal'))
    
    # 3. Beat-aligned internal loops
    segment_beats = feat.beats[
        (feat.beats >= start_frame) & (feat.beats < end_frame)
    ]
    
    n_seg_beats = len(segment_beats)
    if n_seg_beats >= 4:
        # Try different beat multiples
        for n_beats in [4, 8, 16]:
            if n_beats < n_seg_beats:
                for i in range(0, n_seg_beats - n_beats, 2):
                    loop_start = int(segment_beats[i])
                    loop_end = int(segment_beats[i + n_beats])
                    candidates.append((loop_start, loop_end, 'internal'))
    
    return candidates


def _optimize_boundaries(
    feat: Features,
    start_frame: int,
    end_frame: int,
) -> list[tuple[int, int, str]]:
    """
    Optimize the exact boundaries of the segment for seamless looping.
    
    Searches nearby frames for better transition points while
    keeping the loop close to the original selection.
    """
    candidates = []
    
    # Search window (in frames)
    search_window = max(feat.beat_length, 20)
    
    # Candidate start positions
    start_candidates = _find_good_cut_points(
        feat, start_frame, search_window, direction='forward'
    )
    
    # Candidate end positions  
    end_candidates = _find_good_cut_points(
        feat, end_frame, search_window, direction='backward'
    )
    
    # Combine: try each start with each end
    for s in start_candidates[:5]:
        for e in end_candidates[:5]:
            if e > s + feat.beat_length:
                candidates.append((s, e, 'optimized'))
    
    # Also include exact boundaries (user might be precise)
    candidates.append((start_frame, end_frame, 'exact'))
    
    return candidates


def _find_extended_loops(
    feat: Features,
    start_frame: int,
    end_frame: int,
) -> list[tuple[int, int, str]]:
    """
    Allow extension beyond the segment to find better loops.
    
    Sometimes the perfect loop point is just outside the selection.
    """
    candidates = []
    
    # Extension amount
    extension = feat.bar_length  # One bar on each side
    
    extended_start = max(0, start_frame - extension)
    extended_end = min(feat.n_frames, end_frame + extension)
    
    # Get candidates from extended range
    boundary_candidates = _optimize_boundaries(feat, extended_start, extended_end)
    
    for s, e, _ in boundary_candidates:
        candidates.append((s, e, 'extended'))
    
    # Also try phrase-aligned extensions
    for bars_before in range(0, 5):
        for bars_after in range(0, 5):
            s = start_frame - bars_before * feat.bar_length
            e = end_frame + bars_after * feat.bar_length
            
            if s >= 0 and e <= feat.n_frames and e > s + feat.beat_length * 2:
                candidates.append((int(s), int(e), 'extended'))
    
    return candidates


def _find_good_cut_points(
    feat: Features,
    center_frame: int,
    window: int,
    direction: str = 'both',
) -> list[int]:
    """
    Find good frames for cutting near center_frame.
    
    Good cut points have:
    - Low transient strength
    - Beat alignment
    - Low spectral flux
    - Stable dynamics
    """
    results = []
    
    # Search range
    if direction == 'forward':
        search_range = range(center_frame, min(center_frame + window, feat.n_frames))
    elif direction == 'backward':
        search_range = range(max(0, center_frame - window), center_frame + 1)
    else:
        search_range = range(
            max(0, center_frame - window),
            min(center_frame + window, feat.n_frames)
        )
    
    for frame in search_range:
        # Score this frame as a cut point
        trans = feat.transient_strength[frame]
        onset = feat.onset_peaks[frame]
        flux = feat.spectral_flux[frame]
        phase = feat.beat_phase[frame]
        
        # Lower is better for cutting
        trans_score = 1.0 - trans
        onset_score = 1.0 if onset < 0.5 else 0.2
        flux_score = 1.0 - flux
        
        # Prefer on-beat
        phase_score = 1.0 if phase < 0.1 or phase > 0.9 else 0.5
        
        combined = (
            trans_score * 0.35 +
            onset_score * 0.25 +
            flux_score * 0.20 +
            phase_score * 0.20
        )
        
        results.append((combined, frame))
    
    # Sort by score (best first)
    results.sort(reverse=True)
    
    return [frame for _, frame in results]


def _compute_chroma_autocorr(chroma: np.ndarray) -> np.ndarray:
    """Compute autocorrelation of chroma features."""
    n_frames = chroma.shape[1]
    autocorr = np.zeros(n_frames, dtype=np.float32)
    
    # Flatten chroma to 1D for correlation
    chroma_flat = chroma.flatten()
    
    for lag in range(n_frames):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            c1 = chroma[:, :-lag].flatten() if lag < n_frames else np.array([])
            c2 = chroma[:, lag:].flatten()
            
            if len(c1) > 0 and len(c2) > 0:
                n1 = np.linalg.norm(c1)
                n2 = np.linalg.norm(c2)
                if n1 > 1e-10 and n2 > 1e-10:
                    autocorr[lag] = np.dot(c1, c2) / (n1 * n2)
    
    return autocorr


def _generate_explanation(
    ctx: TransitionContext,
    ensemble_score: float,
    crossfade_score: float,
    rhythm_score: float,
    harmonic_score: float,
    boundary_type: str,
) -> str:
    """Generate human-readable explanation of the loop quality."""
    
    parts = []
    
    # Overall quality
    if ensemble_score > 0.8:
        parts.append("Excellent loop quality")
    elif ensemble_score > 0.6:
        parts.append("Good loop quality")
    elif ensemble_score > 0.4:
        parts.append("Acceptable loop quality")
    else:
        parts.append("Marginal loop quality")
    
    # Rhythm
    if rhythm_score > 0.8:
        parts.append("perfect rhythm alignment")
    elif rhythm_score > 0.6:
        parts.append("good rhythm alignment")
    else:
        parts.append("check rhythm at transition")
    
    # Harmony
    if harmonic_score > 0.8:
        parts.append("seamless harmony")
    elif harmonic_score > 0.6:
        parts.append("compatible harmony")
    else:
        parts.append("harmony shift at loop")
    
    # Crossfade
    if crossfade_score > 0.8:
        parts.append("inaudible crossfade")
    elif crossfade_score > 0.6:
        parts.append("smooth crossfade")
    else:
        parts.append("audible crossfade")
    
    # Boundary type
    if boundary_type == 'exact':
        parts.append("at exact selection")
    elif boundary_type == 'optimized':
        parts.append("optimized boundaries")
    else:
        parts.append("extended beyond selection")
    
    return "; ".join(parts)


def _deduplicate_results(
    results: list[SegmentLoopResult],
    min_distance_frames: int = 10,
) -> list[SegmentLoopResult]:
    """Remove results that are too similar to higher-scored ones."""
    if not results:
        return []
    
    deduped = [results[0]]
    
    for result in results[1:]:
        is_duplicate = False
        
        for kept in deduped:
            start_dist = abs(result.start_frame - kept.start_frame)
            end_dist = abs(result.end_frame - kept.end_frame)
            
            if start_dist < min_distance_frames and end_dist < min_distance_frames:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduped.append(result)
    
    return deduped


def find_repetitions_in_segment(
    feat: Features,
    segment_start_frame: int,
    segment_end_frame: int,
    min_repetition_length: int | None = None,
) -> list[tuple[int, int, float]]:
    """
    Find all repeating patterns within a segment.
    
    Returns list of (offset, length, similarity) for each found repetition.
    Useful for understanding the internal structure of the segment.
    """
    segment_length = segment_end_frame - segment_start_frame
    
    if min_repetition_length is None:
        min_repetition_length = max(feat.beat_length * 2, 20)
    
    repetitions = []
    
    # Get segment features
    chroma = feat.chroma_cens[:, segment_start_frame:segment_end_frame]
    
    # Check different offset lengths
    for offset in range(min_repetition_length, segment_length // 2):
        # Compare first part with offset part
        part1 = chroma[:, :segment_length - offset]
        part2 = chroma[:, offset:]
        
        min_len = min(part1.shape[1], part2.shape[1])
        if min_len < min_repetition_length:
            continue
        
        # Compute frame-by-frame similarity
        similarities = np.zeros(min_len)
        for i in range(min_len):
            c1 = part1[:, i]
            c2 = part2[:, i]
            n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
            if n1 > 1e-10 and n2 > 1e-10:
                similarities[i] = np.dot(c1, c2) / (n1 * n2)
        
        # Average similarity for this offset
        avg_sim = np.mean(similarities)
        
        if avg_sim > 0.6:  # Threshold for repetition
            repetitions.append((offset, min_len, avg_sim))
    
    # Sort by similarity
    repetitions.sort(key=lambda x: x[2], reverse=True)
    
    return repetitions

