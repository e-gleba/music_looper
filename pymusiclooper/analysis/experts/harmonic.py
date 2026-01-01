"""
Harmonic Expert - Tonal and Harmonic Transition Quality.

Evaluates:
- Chroma similarity (pitch class distribution)
- Key detection and key distance
- Tonnetz continuity (tonal space)
- Circle of fifths relationships
- Chord progression coherence
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, cosine_similarity


# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

# Circle of fifths distance lookup
FIFTH_DISTANCE = np.array([0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5], dtype=np.int32)


class HarmonicExpert(Expert):
    """Expert for harmonic/tonal transition quality."""
    
    name = "harmonic"
    weight = 0.15
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate harmonic continuity at transition point."""
        
        # 1. Direct chroma similarity
        chroma_sim = cosine_similarity(ctx.chroma_start, ctx.chroma_end)
        
        # 2. Context chroma similarity (more robust)
        context_sim = cosine_similarity(ctx.chroma_context_start, ctx.chroma_context_end)
        
        # 3. Tonnetz continuity (tonal space distance)
        tonnetz_dist = np.linalg.norm(ctx.tonnetz_start - ctx.tonnetz_end)
        tonnetz_sim = max(0, 1.0 - tonnetz_dist / 2.0)
        
        # 4. Key detection and distance
        key_start, mode_start = detect_key(ctx.chroma_context_start)
        key_end, mode_end = detect_key(ctx.chroma_context_end)
        key_distance = compute_key_distance(key_start, mode_start, key_end, mode_end)
        key_match = 1.0 - key_distance
        
        # 5. Harmonic tension (dissonant intervals)
        tension = compute_harmonic_tension(ctx.chroma_start, ctx.chroma_end)
        low_tension = 1.0 - tension
        
        # Weighted combination
        score = (
            chroma_sim * 0.25 +
            context_sim * 0.25 +
            tonnetz_sim * 0.15 +
            key_match * 0.25 +
            low_tension * 0.10
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        chroma_sim = cosine_similarity(ctx.chroma_start, ctx.chroma_end)
        key_start, mode_start = detect_key(ctx.chroma_context_start)
        key_end, mode_end = detect_key(ctx.chroma_context_end)
        
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        mode_names = ['major', 'minor']
        
        return (
            f"Harmonic: {score:.2f} | "
            f"Chroma sim: {chroma_sim:.2f} | "
            f"Key: {key_names[key_start]} {mode_names[mode_start]} → "
            f"{key_names[key_end]} {mode_names[mode_end]}"
        )


@njit(cache=True, fastmath=True)
def detect_key(chroma: np.ndarray) -> tuple[int, int]:
    """Detect key using Krumhansl-Kessler profiles. Returns (key, mode)."""
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    best_key = 0
    best_mode = 0
    best_corr = -2.0
    
    for key in range(12):
        # Rotate chroma to key
        rotated = np.zeros(12)
        for i in range(12):
            rotated[i] = chroma[(i + key) % 12]
        
        # Correlate with profiles
        maj_corr = 0.0
        min_corr = 0.0
        for i in range(12):
            maj_corr += rotated[i] * major_profile[i]
            min_corr += rotated[i] * minor_profile[i]
        
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = key
            best_mode = 0
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = key
            best_mode = 1
    
    return best_key, best_mode


@njit(cache=True, fastmath=True)
def compute_key_distance(key1: int, mode1: int, key2: int, mode2: int) -> float:
    """Compute perceptual distance between keys on circle of fifths."""
    # Circle of fifths distance (0-6)
    fifth_dist = np.array([0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5])
    
    interval = abs(key1 - key2)
    if interval > 6:
        interval = 12 - interval
    cof_dist = fifth_dist[interval]
    
    # Mode penalty
    mode_penalty = 0.0 if mode1 == mode2 else 0.15
    
    # Relative major/minor bonus (e.g., C major ↔ A minor)
    if mode1 != mode2:
        rel_key = (key1 + 3) % 12 if mode1 == 0 else (key1 - 3 + 12) % 12
        if rel_key == key2:
            mode_penalty = 0.05
    
    return min(1.0, cof_dist / 6.0 + mode_penalty)


@njit(cache=True, fastmath=True)
def compute_harmonic_tension(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """Compute harmonic tension based on interval relationships."""
    # Find dominant pitches
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
    
    # Interval tension (dissonance map)
    interval = abs(max1_idx - max2_idx)
    if interval > 6:
        interval = 12 - interval
    
    # Tension values: 0=unison(0), 1=m2(0.9), 2=M2(0.6), 3=m3(0.3), 4=M3(0.25), 5=P4(0.1), 6=tritone(0.85)
    tension_map = np.array([0.0, 0.9, 0.6, 0.3, 0.25, 0.1, 0.85])
    base_tension = tension_map[interval]
    
    # Modulate by chroma similarity
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in range(12):
        dot += chroma1[i] * chroma2[i]
        n1 += chroma1[i] * chroma1[i]
        n2 += chroma2[i] * chroma2[i]
    
    denom = np.sqrt(n1 * n2)
    chroma_sim = dot / denom if denom > 1e-10 else 0.5
    
    return base_tension * 0.6 + (1.0 - chroma_sim) * 0.4

