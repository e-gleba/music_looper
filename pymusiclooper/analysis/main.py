"""
Main Loop Detection Entry Point.

Orchestrates the complete loop detection pipeline:
1. Feature extraction
2. Candidate generation
3. Scoring with expert ensemble
4. Optimization and finalization
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymusiclooper.audio import MLAudio

from pymusiclooper.analysis.features import Features, extract_features
from pymusiclooper.analysis.candidates import LoopPair, find_candidates, prune_candidates
from pymusiclooper.analysis.scoring import score_candidates, finalize_with_alignment, optimize_selection
from pymusiclooper.exceptions import LoopNotFoundError


def find_best_loop_points(
    mlaudio: MLAudio,
    min_duration_multiplier: float = 0.35,
    min_loop_duration: float | None = None,
    max_loop_duration: float | None = None,
    approx_loop_start: float | None = None,
    approx_loop_end: float | None = None,
    brute_force: bool = False,
    disable_pruning: bool = False,
) -> list[LoopPair]:
    """
    Find the best loop points for seamless music looping.
    
    This is the main entry point that orchestrates the complete pipeline:
    
    1. **Feature Extraction**: Extracts spectral, harmonic, rhythmic, and
       perceptual features using librosa and custom algorithms.
    
    2. **Candidate Generation**: Finds potential loop points using
       self-similarity matrices, beat alignment, and phrase structure.
    
    3. **Expert Scoring**: Evaluates each candidate using an 8-expert
       ensemble covering harmonic, rhythmic, timbral, psychoacoustic,
       neuroacoustic, dynamics, onset, and phrase aspects.
    
    4. **Finalization**: Aligns loop points to sample-accurate zero
       crossings for click-free playback.
    
    Args:
        mlaudio: Audio object with loaded audio data
        min_duration_multiplier: Minimum loop duration as fraction of total
        min_loop_duration: Minimum loop duration in seconds (overrides multiplier)
        max_loop_duration: Maximum loop duration in seconds
        approx_loop_start: Approximate loop start in seconds (for guided search)
        approx_loop_end: Approximate loop end in seconds (for guided search)
        brute_force: Check every frame (slower but more thorough)
        disable_pruning: Keep all candidates (for debugging)
    
    Returns:
        List of LoopPair objects sorted by score (best first)
    
    Raises:
        LoopNotFoundError: If no suitable loop points can be found
    """
    t0 = time.perf_counter()
    
    # Calculate frame constraints
    min_frames = (
        mlaudio.seconds_to_frames(min_loop_duration)
        if min_loop_duration else
        mlaudio.seconds_to_frames(int(min_duration_multiplier * mlaudio.total_duration))
    )
    min_frames = max(1, min_frames)
    
    max_frames = (
        mlaudio.seconds_to_frames(max_loop_duration)
        if max_loop_duration else
        mlaudio.seconds_to_frames(mlaudio.total_duration)
    )
    
    # ===== PHASE 1: Feature Extraction =====
    logging.info(f"Extracting features from {mlaudio.filename}...")
    feat = extract_features(
        mlaudio, 
        approx_loop_start, 
        approx_loop_end, 
        brute_force
    )
    logging.info(f"Feature extraction: {time.perf_counter() - t0:.3f}s")
    
    # ===== PHASE 2: Candidate Generation =====
    t1 = time.perf_counter()
    candidates = find_candidates(feat, min_frames, max_frames)
    logging.info(f"Found {len(candidates)} candidates in {time.perf_counter() - t1:.3f}s")
    
    if not candidates:
        raise LoopNotFoundError(f'No loop candidates found for "{mlaudio.filename}".')
    
    # ===== PHASE 3: Scoring =====
    t2 = time.perf_counter()
    scored = score_candidates(mlaudio, feat, candidates, disable_pruning)
    logging.info(f"Scoring completed in {time.perf_counter() - t2:.3f}s")
    
    if not scored:
        raise LoopNotFoundError(f'No loops found for "{mlaudio.filename}".')
    
    # Sort by score (descending)
    scored.sort(key=lambda x: x.score, reverse=True)
    
    # ===== PHASE 4: Optimization =====
    if len(scored) > 1:
        optimize_selection(scored, feat)
    
    # ===== PHASE 5: Finalization =====
    finalize_with_alignment(mlaudio, feat, scored)
    
    if not scored:
        raise LoopNotFoundError(f'No loops found for "{mlaudio.filename}".')
    
    # Final sort
    scored.sort(key=lambda x: x.score, reverse=True)
    
    logging.info(f"Total: {len(scored)} loops in {time.perf_counter() - t0:.3f}s")
    logging.info(f"Best score: {scored[0].score:.4f}")
    
    return scored


def find_loop_for_segment(
    mlaudio: MLAudio,
    segment_start_sec: float,
    segment_end_sec: float,
    search_mode: str = 'boundary',
    n_results: int = 10,
) -> list:
    """
    Find optimal loop points for a user-selected segment.
    
    This is for when the user has selected a specific portion of the audio
    they want to loop. The algorithm finds the best way to make it seamless.
    
    Args:
        mlaudio: Audio object
        segment_start_sec: Start of selection in seconds
        segment_end_sec: End of selection in seconds
        search_mode: 
            - 'internal': Find loops within the segment
            - 'boundary': Optimize start/end for seamless loop
            - 'extended': Allow extending beyond selection
        n_results: Number of results to return
    
    Returns:
        List of SegmentLoopResult objects sorted by score
    """
    from pymusiclooper.analysis.segment_loop import find_segment_loop_points
    
    # Extract features for the full audio (needed for context)
    feat = extract_features(mlaudio, None, None, False)
    
    return find_segment_loop_points(
        mlaudio,
        feat,
        segment_start_sec,
        segment_end_sec,
        search_mode=search_mode,
        n_results=n_results,
    )


# Alias for backwards compatibility
def nearest_zero_crossing(audio, sr, target):
    """Find nearest zero crossing (backwards compatible)."""
    from pymusiclooper.analysis.features import nearest_zero_crossing as nzc
    return nzc(audio, sr, target)

