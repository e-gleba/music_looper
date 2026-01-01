"""
Melodic Expert - Melodic Continuity and Contour Analysis.

Advanced melodic analysis:
- Pitch salience extraction (predominant melody)
- Melodic contour matching (shape similarity)
- Interval preservation
- Melodic phrase detection
- Voice leading analysis
- Pitch stability at transitions
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, cosine_similarity


class MelodicExpert(Expert):
    """
    Expert for melodic continuity and contour preservation.
    
    Focuses on ensuring the melody flows naturally across the loop point
    without abrupt pitch jumps or unnatural interval changes.
    """
    
    name = "melodic"
    weight = 0.14
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate melodic continuity at transition."""
        
        # 1. Pitch salience match (dominant pitch continuity)
        pitch_salience_start = extract_pitch_salience(ctx.chroma_start)
        pitch_salience_end = extract_pitch_salience(ctx.chroma_end)
        
        # Dominant pitch should match or have consonant relationship
        pitch_match = score_pitch_relationship(pitch_salience_start, pitch_salience_end)
        
        # 2. Melodic contour similarity (shape of melody)
        contour_sim = score_contour_similarity(ctx.chroma_start, ctx.chroma_end)
        
        # 3. Interval structure preservation
        interval_struct_start = compute_interval_structure(ctx.chroma_start)
        interval_struct_end = compute_interval_structure(ctx.chroma_end)
        interval_match = 1.0 - np.mean(np.abs(interval_struct_start - interval_struct_end))
        
        # 4. Voice leading quality (smooth voice motion)
        voice_leading = score_voice_leading(ctx.chroma_start, ctx.chroma_end)
        
        # 5. Pitch stability at cut point
        pitch_stability = score_pitch_stability(ctx.chroma_start, ctx.chroma_end)
        
        # 6. Melodic tension/resolution
        melodic_tension = compute_melodic_tension(ctx.chroma_start, ctx.chroma_end)
        low_tension = 1.0 - melodic_tension
        
        # Combine scores
        score = (
            pitch_match * 0.25 +
            contour_sim * 0.20 +
            interval_match * 0.15 +
            voice_leading * 0.20 +
            pitch_stability * 0.10 +
            low_tension * 0.10
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        pitch_start = extract_pitch_salience(ctx.chroma_start)
        pitch_end = extract_pitch_salience(ctx.chroma_end)
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        start_note = note_names[pitch_start[0]] if len(pitch_start) > 0 else "?"
        end_note = note_names[pitch_end[0]] if len(pitch_end) > 0 else "?"
        
        return (
            f"Melodic: {score:.2f} | "
            f"Pitch: {start_note} → {end_note} | "
            f"Contour sim: {score_contour_similarity(ctx.chroma_start, ctx.chroma_end):.2f}"
        )


@njit(cache=True, fastmath=True)
def extract_pitch_salience(chroma: np.ndarray) -> np.ndarray:
    """
    Extract salient pitches (melodic notes) from chroma.
    
    Returns indices of the most prominent pitch classes, ordered by salience.
    """
    # Sort by energy
    indices = np.argsort(chroma)[::-1]
    
    # Return top salient pitches
    n_salient = 3
    salient = np.zeros(n_salient, dtype=np.int64)
    
    count = 0
    for i in range(12):
        if count >= n_salient:
            break
        idx = indices[i]
        if chroma[idx] > 0.1:  # Threshold for salience
            salient[count] = idx
            count += 1
    
    return salient[:count] if count > 0 else np.array([indices[0]])


@njit(cache=True, fastmath=True)
def score_pitch_relationship(pitch1: np.ndarray, pitch2: np.ndarray) -> float:
    """Score melodic relationship between dominant pitches."""
    if len(pitch1) == 0 or len(pitch2) == 0:
        return 0.5
    
    p1 = pitch1[0]
    p2 = pitch2[0]
    
    # Interval between dominant pitches
    interval = abs(p1 - p2)
    if interval > 6:
        interval = 12 - interval
    
    # Consonance by interval
    # 0=unison(1.0), 1=m2(0.3), 2=M2(0.5), 3=m3(0.8), 4=M3(0.85), 5=P4(0.9), 6=tritone(0.4)
    consonance_map = np.array([1.0, 0.3, 0.5, 0.8, 0.85, 0.9, 0.4])
    base_score = consonance_map[interval]
    
    # Bonus if secondary pitches also match
    match_bonus = 0.0
    if len(pitch1) > 1 and len(pitch2) > 1:
        for i in range(1, min(len(pitch1), len(pitch2))):
            if pitch1[i] == pitch2[i]:
                match_bonus += 0.05
    
    return min(1.0, base_score + match_bonus)


def score_contour_similarity(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """
    Score melodic contour similarity.
    
    Contour = the "shape" of the melody (up/down movements).
    Similar contours make transitions more natural.
    """
    # Compute pitch-class "gravity centers"
    weighted_pitch1 = np.sum(chroma1 * np.arange(12)) / (np.sum(chroma1) + 1e-10)
    weighted_pitch2 = np.sum(chroma2 * np.arange(12)) / (np.sum(chroma2) + 1e-10)
    
    # Pitch height difference
    height_diff = abs(weighted_pitch1 - weighted_pitch2)
    height_score = max(0, 1.0 - height_diff / 6)  # 6 semitones = max difference
    
    # Energy distribution similarity (contour shape)
    contour1 = np.gradient(chroma1)
    contour2 = np.gradient(chroma2)
    
    contour_corr = np.corrcoef(contour1, contour2)[0, 1]
    contour_score = (contour_corr + 1) / 2 if not np.isnan(contour_corr) else 0.5
    
    return height_score * 0.5 + contour_score * 0.5


@njit(cache=True, fastmath=True)
def compute_interval_structure(chroma: np.ndarray) -> np.ndarray:
    """
    Compute interval structure vector.
    
    Describes the intervallic relationships between all pitch classes.
    """
    intervals = np.zeros(12, dtype=np.float64)
    
    for i in range(12):
        for j in range(i + 1, 12):
            interval = j - i
            weight = chroma[i] * chroma[j]
            intervals[interval] += weight
    
    # Normalize
    total = np.sum(intervals) + 1e-10
    return intervals / total


@njit(cache=True, fastmath=True)
def score_voice_leading(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """
    Score voice leading quality (smooth pitch motion).
    
    Good voice leading: voices move by small intervals.
    """
    # Find active voices (prominent pitches)
    threshold = 0.2
    
    voices1 = []
    voices2 = []
    
    for i in range(12):
        if chroma1[i] > threshold:
            voices1.append(i)
        if chroma2[i] > threshold:
            voices2.append(i)
    
    if len(voices1) == 0 or len(voices2) == 0:
        return 0.5
    
    # Compute minimum voice motion
    total_motion = 0.0
    matched = 0
    
    for v1 in voices1:
        min_dist = 12
        for v2 in voices2:
            dist = abs(v1 - v2)
            if dist > 6:
                dist = 12 - dist
            if dist < min_dist:
                min_dist = dist
        total_motion += min_dist
        matched += 1
    
    if matched == 0:
        return 0.5
    
    avg_motion = total_motion / matched
    
    # Small motion is good (0=perfect, 6=worst)
    return max(0, 1.0 - avg_motion / 4)


@njit(cache=True, fastmath=True)
def score_pitch_stability(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """
    Score pitch stability at the transition.
    
    Stable pitches (sustained notes) help mask transitions.
    """
    # Find common prominent pitches
    threshold = 0.3
    common = 0.0
    total = 0.0
    
    for i in range(12):
        if chroma1[i] > threshold or chroma2[i] > threshold:
            total += 1
            if chroma1[i] > threshold and chroma2[i] > threshold:
                common += 1
    
    if total == 0:
        return 0.5
    
    return common / total


@njit(cache=True, fastmath=True)
def compute_melodic_tension(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """
    Compute melodic tension at transition.
    
    High tension = unstable melodic motion.
    """
    # Find dominant pitch movement
    max1_idx = 0
    max2_idx = 0
    max1_val = chroma1[0]
    max2_val = chroma2[0]
    
    for i in range(1, 12):
        if chroma1[i] > max1_val:
            max1_val = chroma1[i]
            max1_idx = i
        if chroma2[i] > max2_val:
            max2_val = chroma2[i]
            max2_idx = i
    
    interval = abs(max1_idx - max2_idx)
    if interval > 6:
        interval = 12 - interval
    
    # Tension by interval (dissonant intervals = high tension)
    # m2(1), tritone(6) = high tension; unison, fifths = low tension
    tension_map = np.array([0.0, 0.9, 0.4, 0.25, 0.2, 0.1, 0.8])
    
    return tension_map[interval]


def compute_melodicity_score(features) -> np.ndarray:
    """
    Compute frame-by-frame melodicity score.
    
    High melodicity = clear melodic content (not just chords/noise).
    """
    n_frames = features.n_frames
    melodicity = np.zeros(n_frames, dtype=np.float32)
    
    for i in range(n_frames):
        chroma = features.chroma_cens[:, i]
        
        # 1. Pitch clarity (one dominant pitch vs. spread)
        sorted_chroma = np.sort(chroma)[::-1]
        if sorted_chroma[0] > 0:
            clarity = sorted_chroma[0] / (np.sum(sorted_chroma[:3]) + 1e-10)
        else:
            clarity = 0.0
        
        # 2. Spectral stability (low flux = held notes)
        flux = features.spectral_flux[i]
        stability = 1.0 - flux
        
        # 3. Harmonic content (low flatness = tonal)
        tonality = 1.0 - features.spectral_flatness[i]
        
        melodicity[i] = clarity * 0.4 + stability * 0.3 + tonality * 0.3
    
    return melodicity


def score_melodic_continuity(
    features,
    start_frame: int,
    end_frame: int,
    context_frames: int = 10,
) -> float:
    """
    Score melodic continuity across a transition.
    
    Comprehensive melodic analysis including contour, intervals, and pitch flow.
    """
    n_f = features.n_frames
    b1 = min(max(0, start_frame), n_f - 1)
    b2 = min(max(0, end_frame - 1), n_f - 1)
    ctx = min(context_frames, b1, n_f - b2 - 1)
    
    # Get context windows
    pre_chroma = features.chroma_cens[:, max(0, b2 - ctx):b2 + 1]
    post_chroma = features.chroma_cens[:, b1:min(b1 + ctx + 1, n_f)]
    
    if pre_chroma.shape[1] < 2 or post_chroma.shape[1] < 2:
        return 0.5
    
    # 1. Pitch trajectory continuity
    pre_trajectory = np.sum(pre_chroma * np.arange(12).reshape(-1, 1), axis=0)
    pre_trajectory /= np.sum(pre_chroma, axis=0) + 1e-10
    
    post_trajectory = np.sum(post_chroma * np.arange(12).reshape(-1, 1), axis=0)
    post_trajectory /= np.sum(post_chroma, axis=0) + 1e-10
    
    # Check if trajectories connect smoothly
    pre_end = pre_trajectory[-1]
    post_start = post_trajectory[0]
    trajectory_jump = abs(pre_end - post_start)
    trajectory_score = max(0, 1.0 - trajectory_jump / 6)
    
    # 2. Contour correlation
    if len(pre_trajectory) > 2 and len(post_trajectory) > 2:
        pre_contour = np.diff(pre_trajectory)
        post_contour = np.diff(post_trajectory)
        
        # Pad to same length
        min_len = min(len(pre_contour), len(post_contour))
        if min_len > 1:
            corr = np.corrcoef(pre_contour[-min_len:], post_contour[:min_len])[0, 1]
            contour_corr = (corr + 1) / 2 if not np.isnan(corr) else 0.5
        else:
            contour_corr = 0.5
    else:
        contour_corr = 0.5
    
    # 3. Interval structure consistency
    pre_mean_chroma = np.mean(pre_chroma, axis=1)
    post_mean_chroma = np.mean(post_chroma, axis=1)
    
    pre_intervals = compute_interval_structure(pre_mean_chroma)
    post_intervals = compute_interval_structure(post_mean_chroma)
    
    interval_sim = cosine_similarity(pre_intervals, post_intervals)
    
    return trajectory_score * 0.4 + contour_corr * 0.3 + interval_sim * 0.3

