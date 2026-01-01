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


def detect_key(chroma: np.ndarray) -> tuple[int, int]:
    """Detect key using Krumhansl-Kessler profiles. Returns (key, mode)."""
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Vectorized: rotate chroma for all 12 keys at once
    keys = np.arange(12)
    indices = (np.arange(12)[:, np.newaxis] + keys) % 12  # Shape: (12, 12)
    rotated_chromas = chroma[indices]  # Shape: (12, 12)
    
    # Vectorized correlation: (12, 12) @ (12,) = (12,)
    maj_corrs = rotated_chromas @ major_profile
    min_corrs = rotated_chromas @ minor_profile
    
    # Find best correlation
    all_corrs = np.concatenate([maj_corrs, min_corrs])
    best_idx = np.argmax(all_corrs)
    
    if best_idx < 12:
        return int(best_idx), 0  # Major
    else:
        return int(best_idx - 12), 1  # Minor


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


def compute_harmonic_tension(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """Compute harmonic tension based on interval relationships."""
    # Vectorized: find dominant pitches
    max1_idx = np.argmax(chroma1)
    max2_idx = np.argmax(chroma2)
    
    # Interval tension (dissonance map)
    interval = abs(max1_idx - max2_idx)
    if interval > 6:
        interval = 12 - interval
    
    # Tension values: 0=unison(0), 1=m2(0.9), 2=M2(0.6), 3=m3(0.3), 4=M3(0.25), 5=P4(0.1), 6=tritone(0.85)
    tension_map = np.array([0.0, 0.9, 0.6, 0.3, 0.25, 0.1, 0.85])
    base_tension = tension_map[interval]
    
    # Vectorized chroma similarity (cosine similarity)
    dot = np.dot(chroma1, chroma2)
    n1 = np.linalg.norm(chroma1)
    n2 = np.linalg.norm(chroma2)
    denom = n1 * n2
    chroma_sim = dot / denom if denom > 1e-10 else 0.5
    
    return base_tension * 0.6 + (1.0 - chroma_sim) * 0.4

