"""
Loop Candidate Generation Module.

Generates potential loop point candidates using multiple strategies:
- Self-similarity matrix (SSM) based detection
- Beat-aligned candidates
- Downbeat-aligned candidates
- Phrase-structure aware candidates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import uniform_filter1d

if TYPE_CHECKING:
    from pymusiclooper.analysis.features import Features

from pymusiclooper.analysis.constants import (
    CHROMA_SIM_MIN,
    MFCC_SIM_MIN,
    MFCC_SIM_ACCEPTABLE,
    COMBINED_SIM_MIN,
    ONSET_SIM_MIN,
    SIM_WEIGHT_CHROMA,
    SIM_WEIGHT_MFCC,
    SIM_WEIGHT_ONSET,
    MFCC_CONTEXT_WINDOW,
    ONSET_CONTEXT_WINDOW,
    compute_adaptive_timbre_threshold
)


@dataclass(slots=True)
class LoopPair:
    """Represents a potential loop point with scoring metadata."""
    _loop_start_frame_idx: int
    _loop_end_frame_idx: int
    note_distance: float      # Harmonic distance (0=identical, 1=different)
    loudness_difference: float  # Energy difference in dB
    score: float = 0.0        # Final composite score
    loop_start: int = 0       # Final sample position
    loop_end: int = 0         # Final sample position


def find_candidates(
    feat: Features,
    min_frames: int,
    max_frames: int,
) -> list[LoopPair]:
    """
    Generate loop candidates using multiple detection strategies.
    
    Combines SSM-based detection with beat/downbeat alignment for robust
    candidate generation across different music styles.
    """
    candidates = []
    existing = set()
    
    # SSM-based candidates (multiple thresholds for robustness)
    for thresh in [45, 55, 65]:
        candidates.extend(_ssm_candidates(
            feat.ssm_chroma, feat.beats, feat.power_db, 
            min_frames, max_frames, feat.n_frames, thresh
        ))
        candidates.extend(_ssm_candidates(
            feat.ssm_mfcc, feat.beats, feat.power_db,
            min_frames, max_frames, feat.n_frames, thresh + 5
        ))
    
    # Structure-aligned candidates
    candidates.extend(_downbeat_candidates(feat, min_frames, max_frames))
    candidates.extend(_beat_candidates(feat, min_frames, max_frames))
    candidates.extend(_phrase_candidates(feat, min_frames, max_frames))
    
    # Deduplicate
    unique = []
    for c in candidates:
        key = (c._loop_start_frame_idx, c._loop_end_frame_idx)
        if key not in existing:
            unique.append(c)
            existing.add(key)
    
    # Final timbre filtering - reject candidates with very different timbre
    # This catches cases like guitar at start but no guitar at end
    # Use adaptive threshold for drums-heavy music
    filtered = []
    avg_transient = np.mean(feat.transient_strength) if len(feat.transient_strength) > 0 else 0.3
    adaptive_threshold = compute_adaptive_timbre_threshold(avg_transient)
    
    for c in unique:
        f1 = min(c._loop_start_frame_idx, feat.n_frames - 1)
        f2 = min(c._loop_end_frame_idx, feat.n_frames - 1)
        
        # Check MFCC similarity in context window
        window = min(MFCC_CONTEXT_WINDOW, feat.n_frames // 100)
        mfcc_start = feat.mfcc[:, max(0, f1-window):min(feat.n_frames, f1+window)]
        mfcc_end = feat.mfcc[:, max(0, f2-window):min(feat.n_frames, f2+window)]
        
        if mfcc_start.shape[1] > 0 and mfcc_end.shape[1] > 0:
            mfcc1_avg = np.mean(mfcc_start, axis=1)
            mfcc2_avg = np.mean(mfcc_end, axis=1)
            mfcc_sim = np.dot(mfcc1_avg, mfcc2_avg) / (np.linalg.norm(mfcc1_avg) * np.linalg.norm(mfcc2_avg) + 1e-10)
            
            # Only keep if timbre is reasonably similar (adaptive threshold)
            if mfcc_sim >= MFCC_SIM_ACCEPTABLE:
                filtered.append(c)
        else:
            # If we can't check, keep it (edge case)
            filtered.append(c)
    
    return filtered


def _ssm_candidates(
    ssm: np.ndarray,
    beats: np.ndarray,
    power_db: np.ndarray,
    min_frames: int,
    max_frames: int,
    n_frames: int,
    threshold_pct: int,
) -> list[LoopPair]:
    """Generate candidates from self-similarity matrix diagonals."""
    n_beats = len(beats)
    if n_beats < 4:
        return []
    
    candidates = []
    power = np.max(power_db, axis=0)
    
    # Upper triangle values for threshold calculation
    ssm_upper = np.triu(ssm, k=1)
    valid_vals = ssm_upper[ssm_upper > 0.05]
    if len(valid_vals) < 10:
        return []
    
    threshold = np.percentile(valid_vals, threshold_pct)
    
    # Process diagonals
    max_offset = min(n_beats, 200)
    
    for offset in range(1, max_offset):
        diag = np.diag(ssm, k=offset)
        if len(diag) < 2:
            continue
        
        # Smooth and threshold
        diag_smooth = uniform_filter1d(diag, size=3)
        above_thresh = diag_smooth >= threshold
        
        if not np.any(above_thresh):
            continue
        
        # Find contiguous regions
        changes = np.diff(above_thresh.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        if above_thresh[0]:
            starts = np.concatenate([[0], starts])
        if above_thresh[-1]:
            ends = np.concatenate([ends, [len(above_thresh)]])
        
        for s, e in zip(starts, ends):
            if e - s < 1:
                continue
            
            beat_start = s
            beat_end = s + offset
            
            if beat_end >= n_beats:
                continue
            
            frame_start = beats[beat_start]
            frame_end = beats[beat_end]
            loop_len = frame_end - frame_start
            
            if loop_len < min_frames or loop_len > max_frames:
                continue
            
            sim = float(np.mean(diag_smooth[s:e]))
            ps = power[min(int(frame_start), len(power) - 1)]
            pe = power[max(0, min(int(frame_end) - 1, len(power) - 1))]
            
            candidates.append(LoopPair(
                _loop_start_frame_idx=int(frame_start),
                _loop_end_frame_idx=int(frame_end),
                note_distance=1.0 - sim,
                loudness_difference=abs(ps - pe),
            ))
    
    return candidates


def _beat_candidates(feat: Features, min_frames: int, max_frames: int) -> list[LoopPair]:
    """Generate beat-aligned candidates."""
    candidates = []
    beats = feat.beats
    n_beats = len(beats)
    power = np.max(feat.power_db, axis=0)
    
    if n_beats < 8:
        return []
    
    step = max(1, n_beats // 150)
    
    for i in range(0, n_beats - 8, step):
        for j in range(i + 8, n_beats, step):
            loop_len = beats[j] - beats[i]
            
            if loop_len < min_frames:
                continue
            if loop_len > max_frames:
                break
            
            # Multi-feature similarity check (chroma + timbre + onset pattern)
            f1 = min(int(beats[i]), feat.n_frames - 1)
            f2 = min(int(beats[j]), feat.n_frames - 1)
            
            # 1. Chroma similarity (harmonic content)
            c1 = feat.chroma_cens[:, f1]
            c2 = feat.chroma_cens[:, f2]
            chroma_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
            
            if chroma_sim < CHROMA_SIM_MIN:
                continue
            
            # 2. MFCC similarity (timbre/instruments) - check context window
            window = min(MFCC_CONTEXT_WINDOW, feat.n_frames // 100)
            mfcc_start = feat.mfcc[:, max(0, f1-window):min(feat.n_frames, f1+window)]
            mfcc_end = feat.mfcc[:, max(0, f2-window):min(feat.n_frames, f2+window)]
            
            if mfcc_start.shape[1] > 0 and mfcc_end.shape[1] > 0:
                # Vectorized: mean along time axis, then cosine similarity
                mfcc1_avg = np.mean(mfcc_start, axis=1)
                mfcc2_avg = np.mean(mfcc_end, axis=1)
                # Use np.dot and np.linalg.norm for vectorized cosine similarity
                mfcc_sim = np.dot(mfcc1_avg, mfcc2_avg) / (np.linalg.norm(mfcc1_avg) * np.linalg.norm(mfcc2_avg) + 1e-10)
            else:
                mfcc_sim = 0.5
            
            # 3. Onset pattern similarity (where notes/instruments start)
            # Check if onset patterns match - if guitar starts at beginning but not at end, this will catch it
            onset_window = min(ONSET_CONTEXT_WINDOW, feat.n_frames // 50)
            onset_start = feat.onset_env[max(0, f1-onset_window):min(feat.n_frames, f1+onset_window)]
            onset_end = feat.onset_env[max(0, f2-onset_window):min(feat.n_frames, f2+onset_window)]
            
            if len(onset_start) > 5 and len(onset_end) > 5:
                # Normalize and compare patterns
                onset_start_norm = (onset_start - np.mean(onset_start)) / (np.std(onset_start) + 1e-10)
                onset_end_norm = (onset_end - np.mean(onset_end)) / (np.std(onset_end) + 1e-10)
                
                min_len = min(len(onset_start_norm), len(onset_end_norm))
                if min_len > 5:
                    onset_corr = np.corrcoef(onset_start_norm[:min_len], onset_end_norm[:min_len])[0, 1]
                    onset_sim = max(0, onset_corr) if not np.isnan(onset_corr) else ONSET_SIM_MIN
                else:
                    onset_sim = 0.5
            else:
                onset_sim = 0.5
            
            # Combined similarity - require reasonable match
            combined_sim = (
                chroma_sim * SIM_WEIGHT_CHROMA + 
                mfcc_sim * SIM_WEIGHT_MFCC + 
                onset_sim * SIM_WEIGHT_ONSET
            )
            
            # Moderate threshold - allow some variation
            if combined_sim < COMBINED_SIM_MIN:
                continue
            
            # Additional check: if timbre is very different, reject
            # Use adaptive threshold based on transient strength (for drums)
            avg_transient = (feat.transient_strength[f1] + feat.transient_strength[f2]) / 2
            adaptive_mfcc_threshold = compute_adaptive_timbre_threshold(avg_transient)
            
            if mfcc_sim < adaptive_mfcc_threshold:
                continue
            
            ps = power[min(int(beats[i]), len(power) - 1)]
            pe = power[max(0, min(int(beats[j]) - 1, len(power) - 1))]
            
            candidates.append(LoopPair(
                _loop_start_frame_idx=int(beats[i]),
                _loop_end_frame_idx=int(beats[j]),
                note_distance=1.0 - combined_sim,  # Use combined_sim instead of undefined sim
                loudness_difference=abs(ps - pe),
            ))
    
    return candidates


def _downbeat_candidates(feat: Features, min_frames: int, max_frames: int) -> list[LoopPair]:
    """Generate downbeat-aligned candidates (musically stronger boundaries)."""
    candidates = []
    downbeats = feat.downbeats
    n_db = len(downbeats)
    power = np.max(feat.power_db, axis=0)
    chroma = feat.chroma_cens
    n_f = feat.n_frames
    
    if n_db < 2:
        return []
    
    for i in range(n_db):
        for j in range(i + 1, n_db):
            loop_len = downbeats[j] - downbeats[i]
            
            if loop_len < min_frames:
                continue
            if loop_len > max_frames:
                break
            
            frame_i = min(int(downbeats[i]), n_f - 1)
            frame_j = max(0, min(int(downbeats[j]) - 1, n_f - 1))
            
            # Chroma similarity
            c1 = chroma[:, frame_i]
            c2 = chroma[:, frame_j]
            sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
            
            ps = power[frame_i]
            pe = power[frame_j]
            
            candidates.append(LoopPair(
                _loop_start_frame_idx=int(downbeats[i]),
                _loop_end_frame_idx=int(downbeats[j]),
                note_distance=1.0 - sim,
                loudness_difference=abs(ps - pe),
            ))
    
    return candidates


def _phrase_candidates(feat: Features, min_frames: int, max_frames: int) -> list[LoopPair]:
    """Generate candidates at likely phrase boundaries (4, 8, 16 bars)."""
    candidates = []
    
    if feat.bar_length <= 0:
        return []
    
    # Phrase lengths in bars
    phrase_bars = [4, 8, 16, 32]
    power = np.max(feat.power_db, axis=0)
    
    for n_bars in phrase_bars:
        phrase_len = n_bars * feat.bar_length
        
        if phrase_len < min_frames or phrase_len > max_frames:
            continue
        
        # Scan for phrase-aligned loop points
        step = feat.bar_length
        for start in range(0, feat.n_frames - phrase_len, step):
            end = start + phrase_len
            
            if end >= feat.n_frames:
                break
            
            # Check chroma similarity
            c1 = feat.chroma_cens[:, start]
            c2 = feat.chroma_cens[:, min(end, feat.n_frames - 1)]
            sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
            
            if sim < 0.5:  # Stricter threshold for phrase candidates
                continue
            
            ps = power[min(start, len(power) - 1)]
            pe = power[max(0, min(end - 1, len(power) - 1))]
            
            candidates.append(LoopPair(
                _loop_start_frame_idx=start,
                _loop_end_frame_idx=end,
                note_distance=1.0 - sim,
                loudness_difference=abs(ps - pe),
            ))
    
    return candidates


def prune_candidates(candidates: list[LoopPair], max_keep: int = 100) -> list[LoopPair]:
    """Prune candidates to keep the most promising ones."""
    if len(candidates) <= max_keep:
        return candidates
    
    # Sort by combined heuristic (low note_distance and loudness_difference is good)
    candidates.sort(key=lambda x: (x.note_distance * 0.7 + x.loudness_difference * 0.3))
    
    return candidates[:max_keep]

