"""
Multi-hop loop detection - finding similar sections (choruses, leitmotifs).

This module builds on top of the base analysis to find repeating patterns
in different parts of the song and create jump chains between them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymusiclooper.audio import MLAudio
    from pymusiclooper.analysis import Features

from pymusiclooper.analysis import nearest_zero_crossing


@dataclass
class MultiHopLoop:
    """Multi-hop loop chain."""
    segments: list[tuple[int, int]]  # List of (start_frame, end_frame)
    transitions: list[float]  # Transition quality scores
    total_score: float = 0.0
    segment_samples: list[tuple[int, int]] = field(default_factory=list)  # Sample positions


def find_multi_hop_loops(
    mlaudio: MLAudio,
    features: Features,
    n_hops: int = 2,
    min_segment_duration: float = 4.0,
    max_segment_duration: float | None = None,
) -> list[MultiHopLoop]:
    """
    Find multi-hop loop chains between SIMILAR SECTIONS (choruses, leitmotifs).
    
    Finds repeating patterns in different parts of the song and creates
    jump chains between them.
    
    Example with n_hops=2:
    - Segment 1: 00:10-00:18 (first chorus)
    - Segment 2: 01:30-01:38 (second chorus, similar!)
    - Jump back to Segment 1
    
    Args:
        mlaudio: Audio object
        features: Pre-extracted features from base analysis
        n_hops: Number of similar sections to chain (2, 3, etc.)
        min_segment_duration: Minimum segment length in seconds
        max_segment_duration: Maximum segment length in seconds
    
    Returns:
        List of MultiHopLoop chains with NON-ADJACENT similar segments
    """
    t0 = time.perf_counter()
    
    min_frames = mlaudio.seconds_to_frames(min_segment_duration)
    max_frames = (
        mlaudio.seconds_to_frames(max_segment_duration)
        if max_segment_duration else
        mlaudio.seconds_to_frames(min(30.0, mlaudio.total_duration / 3))
    )
    
    # Minimum gap between segments (they should be in DIFFERENT parts of song!)
    min_gap_seconds = max(3.0, min_segment_duration * 1.5)
    min_gap_frames = mlaudio.seconds_to_frames(min_gap_seconds)
    
    logging.info(f"Multi-hop analysis: n_hops={n_hops}, min_seg={min_segment_duration:.1f}s, min_gap={min_gap_seconds:.1f}s")
    
    # Find similar sections using features
    chains = _find_similar_sections(features, n_hops, min_frames, max_frames, min_gap_frames)
    
    if not chains:
        logging.warning("No similar sections found")
        return []
    
    # Finalize sample positions with rhythm-preserving alignment
    for chain in chains:
        chain.segment_samples = []
        for start_f, end_f in chain.segments:
            # Get beat phases and transient strengths to preserve rhythm alignment
            start_phase = float(features.beat_phase[min(int(start_f), len(features.beat_phase) - 1)])
            end_phase = float(features.beat_phase[min(int(end_f), len(features.beat_phase) - 1)])
            start_transient = float(features.transient_strength[min(int(start_f), len(features.transient_strength) - 1)])
            end_transient = float(features.transient_strength[min(int(end_f), len(features.transient_strength) - 1)])
            
            start_s = nearest_zero_crossing(
                mlaudio.playback_audio, mlaudio.rate,
                mlaudio.frames_to_samples(start_f),
                beat_phase=start_phase,
                transient_strength=start_transient
            )
            end_s = nearest_zero_crossing(
                mlaudio.playback_audio, mlaudio.rate,
                mlaudio.frames_to_samples(end_f),
                beat_phase=end_phase,
                transient_strength=end_transient
            )
            chain.segment_samples.append((start_s, end_s))
    
    chains.sort(key=lambda x: x.total_score, reverse=True)
    
    logging.info(f"Found {len(chains)} similar section chains in {time.perf_counter() - t0:.3f}s")
    if chains:
        logging.info(f"Best chain score: {chains[0].total_score:.4f}")
    
    return chains


def _find_similar_sections(
    feat: Features,
    n_hops: int,
    min_frames: int,
    max_frames: int,
    min_gap: int,
) -> list[MultiHopLoop]:
    """Find similar sections (choruses, leitmotifs) in different parts of the song."""
    
    beats = feat.beats
    n_beats = len(beats)
    
    if n_beats < 8:
        return []
    
    chains = []
    
    # Use beats directly with step for efficiency
    step = max(1, n_beats // 200)
    segment_starts = beats[::step]
    n_segments = len(segment_starts)
    
    if n_segments < n_hops + 1:
        return []
    
    # For each possible segment length (in terms of step)
    for seg_len_steps in range(2, min(20, n_segments // 2)):
        seg_len_frames = seg_len_steps * step * (feat.beat_length or 20)
        
        # Check segment length constraints
        if seg_len_frames < min_frames:
            continue
        if seg_len_frames > max_frames:
            continue
        
        # Find all similar segment pairs
        similar_pairs = []
        
        for i in range(n_segments - seg_len_steps):
            seg_i_start = segment_starts[i]
            seg_i_end_idx = min(i + seg_len_steps, n_segments - 1)
            seg_i_end = segment_starts[seg_i_end_idx]
            
            actual_len_i = seg_i_end - seg_i_start
            if actual_len_i < min_frames or actual_len_i > max_frames:
                continue
            
            for j in range(i + 1, n_segments - seg_len_steps):
                seg_j_start = segment_starts[j]
                seg_j_end_idx = min(j + seg_len_steps, n_segments - 1)
                seg_j_end = segment_starts[seg_j_end_idx]
                
                # Must be far apart (different parts of song!)
                if seg_j_start - seg_i_end < min_gap:
                    continue
                
                actual_len_j = seg_j_end - seg_j_start
                if actual_len_j < min_frames or actual_len_j > max_frames:
                    continue
                
                # Calculate similarity between segments
                sim = _segment_similarity(feat, seg_i_start, seg_i_end, seg_j_start, seg_j_end)
                
                if sim > 0.4:  # Relaxed threshold
                    similar_pairs.append((
                        (seg_i_start, seg_i_end),
                        (seg_j_start, seg_j_end),
                        sim
                    ))
        
        # Build chains from similar pairs
        if n_hops == 2 and similar_pairs:
            # Simple 2-hop: play segment A, jump to similar segment B, loop back
            for (seg_a, seg_b, sim) in similar_pairs:
                # Check transition quality (end of A to start of B, and end of B back to start of A)
                trans_ab = _transition_quality(feat, seg_a[1], seg_b[0])
                trans_ba = _transition_quality(feat, seg_b[1], seg_a[0])
                
                total_score = sim * 0.5 + trans_ab * 0.25 + trans_ba * 0.25
                
                if total_score > 0.35:
                    chains.append(MultiHopLoop(
                        segments=[seg_a, seg_b],
                        transitions=[trans_ab, trans_ba],
                        total_score=float(total_score),
                    ))
        
        elif n_hops >= 3 and len(similar_pairs) >= 2:
            # Find chains of 3+ similar segments by extending 2-hop chains
            for (seg_a, seg_b, sim_ab) in similar_pairs:
                # Find a third segment similar to both
                for (seg_c, seg_d, sim_cd) in similar_pairs:
                    # seg_d should be similar to seg_a or seg_b
                    if seg_d[0] <= seg_b[1] + min_gap:
                        continue
                    
                    # Check similarity
                    sim_bd = _segment_similarity(feat, seg_b[0], seg_b[1], seg_d[0], seg_d[1])
                    if sim_bd < 0.35:
                        continue
                    
                    trans_ab = _transition_quality(feat, seg_a[1], seg_b[0])
                    trans_bd = _transition_quality(feat, seg_b[1], seg_d[0])
                    trans_da = _transition_quality(feat, seg_d[1], seg_a[0])
                    
                    total_score = (sim_ab + sim_bd) / 2 * 0.5 + (trans_ab + trans_bd + trans_da) / 3 * 0.5
                    
                    if total_score > 0.3:
                        chains.append(MultiHopLoop(
                            segments=[seg_a, seg_b, seg_d],
                            transitions=[trans_ab, trans_bd, trans_da],
                            total_score=float(total_score),
                        ))
                        
                        if len(chains) > 1000:  # Limit
                            break
                if len(chains) > 1000:
                    break
    
    return chains


def _segment_similarity(feat: Features, s1_start: int, s1_end: int, s2_start: int, s2_end: int) -> float:
    """Calculate similarity between two segments."""
    n_f = feat.n_frames
    
    # Chroma similarity
    c1 = feat.chroma_cens[:, max(0, min(s1_start, n_f-1)):min(s1_end, n_f)]
    c2 = feat.chroma_cens[:, max(0, min(s2_start, n_f-1)):min(s2_end, n_f)]
    
    if c1.shape[1] == 0 or c2.shape[1] == 0:
        return 0.0
    
    # Average chroma of each segment
    c1_avg = np.mean(c1, axis=1)
    c2_avg = np.mean(c2, axis=1)
    
    c1_norm = c1_avg / (np.linalg.norm(c1_avg) + 1e-10)
    c2_norm = c2_avg / (np.linalg.norm(c2_avg) + 1e-10)
    
    chroma_sim = np.dot(c1_norm, c2_norm)
    
    # MFCC similarity (timbral)
    m1 = feat.mfcc[:, max(0, min(s1_start, n_f-1)):min(s1_end, n_f)]
    m2 = feat.mfcc[:, max(0, min(s2_start, n_f-1)):min(s2_end, n_f)]
    
    if m1.shape[1] > 0 and m2.shape[1] > 0:
        m1_avg = np.mean(m1, axis=1)
        m2_avg = np.mean(m2, axis=1)
        m1_norm = m1_avg / (np.linalg.norm(m1_avg) + 1e-10)
        m2_norm = m2_avg / (np.linalg.norm(m2_avg) + 1e-10)
        mfcc_sim = np.dot(m1_norm, m2_norm)
    else:
        mfcc_sim = 0.0
    
    # Energy profile similarity
    e1 = feat.rms[max(0, min(s1_start, len(feat.rms)-1)):min(s1_end, len(feat.rms))]
    e2 = feat.rms[max(0, min(s2_start, len(feat.rms)-1)):min(s2_end, len(feat.rms))]
    
    if len(e1) > 2 and len(e2) > 2:
        # Resample to same length
        target_len = min(len(e1), len(e2), 20)
        e1_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(e1)), e1)
        e2_rs = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(e2)), e2)
        
        # Check variance to avoid warnings
        if np.std(e1_rs) > 1e-8 and np.std(e2_rs) > 1e-8:
            corr = np.corrcoef(e1_rs, e2_rs)[0, 1]
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.0
        energy_sim = (corr + 1) / 2
    else:
        energy_sim = 0.5
    
    return chroma_sim * 0.5 + mfcc_sim * 0.3 + energy_sim * 0.2


def _transition_quality(feat: Features, from_frame: int, to_frame: int) -> float:
    """Quality of jumping from one point to another."""
    n_f = feat.n_frames
    
    f1 = min(from_frame, n_f - 1)
    f2 = min(to_frame, n_f - 1)
    
    # Chroma match at transition
    c1 = feat.chroma_cens[:, f1]
    c2 = feat.chroma_cens[:, f2]
    chroma_sim = 1.0 - np.linalg.norm(c1 - c2) / 2
    
    # Energy match
    e1 = feat.rms[min(f1, len(feat.rms) - 1)]
    e2 = feat.rms[min(f2, len(feat.rms) - 1)]
    energy_match = 1.0 - abs(e1 - e2) * 3
    
    # Low transient at both points = good
    t1 = feat.transient_strength[min(f1, len(feat.transient_strength) - 1)]
    t2 = feat.transient_strength[min(f2, len(feat.transient_strength) - 1)]
    transient_score = 1.0 - (t1 + t2) / 2
    
    return max(0, chroma_sim * 0.4 + energy_match * 0.3 + transient_score * 0.3)

