"""
Neuroacoustic Expert - Neural Processing of Music.

Models higher-level musical cognition:
- Consonance/dissonance perception (Plomp-Levelt, Sethares)
- Roughness (beating between partials)
- Harmonic series alignment
- Virtual pitch
- Musical tension and resolution
- Auditory Gestalt principles
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, cosine_similarity


class NeuroacousticExpert(Expert):
    """
    Expert modeling neural/cognitive processing of musical transitions.
    
    Focuses on:
    - Consonance: How pleasant/stable the harmonic content is
    - Roughness: Beating between nearby frequencies
    - Tension: Musical tension that creates expectation
    - Resolution: Satisfying resolution of tension
    """
    
    name = "neuroacoustic"
    weight = 0.14
    
    def score(self, ctx: TransitionContext) -> float:
        """Score based on neural/cognitive music perception."""
        
        # 1. Consonance at transition points
        cons_start = consonance_model(ctx.chroma_start)
        cons_end = consonance_model(ctx.chroma_end)
        
        # Similar consonance = smooth transition
        cons_match = 1.0 - abs(cons_start - cons_end)
        
        # High consonance at cut point is good
        avg_consonance = (cons_start + cons_end) / 2
        
        # 2. Roughness (sensory dissonance from beating)
        rough_start = roughness_model(ctx.chroma_start)
        rough_end = roughness_model(ctx.chroma_end)
        
        # Low roughness at transition is good
        low_roughness = 1.0 - (rough_start + rough_end) / 2
        
        # Similar roughness = smooth transition
        rough_match = 1.0 - abs(rough_start - rough_end)
        
        # 3. Harmonic tension between endpoints
        tension = harmonic_tension_model(ctx.chroma_start, ctx.chroma_end)
        low_tension = 1.0 - tension
        
        # 4. Spectral harmonicity
        harmonicity_start = estimate_harmonicity(ctx.mfcc_start)
        harmonicity_end = estimate_harmonicity(ctx.mfcc_end)
        harmonicity_match = 1.0 - abs(harmonicity_start - harmonicity_end)
        
        # 5. Gestalt continuity
        # Good continuation: similar features should continue
        gestalt_score = compute_gestalt_continuity(ctx)
        
        # 6. Tension-resolution profile
        # A loop should not cut during high tension expecting resolution
        resolution_score = score_tension_resolution(ctx)
        
        # Combine scores
        score = (
            cons_match * 0.18 +
            avg_consonance * 0.12 +
            low_roughness * 0.15 +
            rough_match * 0.10 +
            low_tension * 0.18 +
            harmonicity_match * 0.10 +
            gestalt_score * 0.10 +
            resolution_score * 0.07
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        cons = (consonance_model(ctx.chroma_start) + consonance_model(ctx.chroma_end)) / 2
        rough = (roughness_model(ctx.chroma_start) + roughness_model(ctx.chroma_end)) / 2
        tension = harmonic_tension_model(ctx.chroma_start, ctx.chroma_end)
        
        return (
            f"Neuroacoustic: {score:.2f} | "
            f"Consonance: {cons:.2f} | "
            f"Roughness: {rough:.2f} | "
            f"Tension: {tension:.2f}"
        )


@njit(cache=True, fastmath=True)
def consonance_model(chroma: np.ndarray) -> float:
    """
    Compute consonance based on harmonic series and Pythagorean ratios.
    
    Based on Plomp & Levelt (1965) and modern extensions.
    Intervals considered consonant: unison, octave, fifth, fourth, major/minor third.
    """
    consonance = 0.0
    total_weight = 0.0
    
    # Self-consonance (pure tones)
    for i in range(12):
        consonance += chroma[i] * chroma[i] * 1.0
        total_weight += chroma[i] * 1.0
    
    # Perfect fifth (7 semitones) - most consonant after unison/octave
    for i in range(12):
        j = (i + 7) % 12
        consonance += chroma[i] * chroma[j] * 0.9
        total_weight += (chroma[i] + chroma[j]) * 0.45
    
    # Perfect fourth (5 semitones)
    for i in range(12):
        j = (i + 5) % 12
        consonance += chroma[i] * chroma[j] * 0.85
        total_weight += (chroma[i] + chroma[j]) * 0.425
    
    # Major third (4 semitones)
    for i in range(12):
        j = (i + 4) % 12
        consonance += chroma[i] * chroma[j] * 0.75
        total_weight += (chroma[i] + chroma[j]) * 0.375
    
    # Minor third (3 semitones)
    for i in range(12):
        j = (i + 3) % 12
        consonance += chroma[i] * chroma[j] * 0.7
        total_weight += (chroma[i] + chroma[j]) * 0.35
    
    if total_weight < 1e-6:
        return 0.5
    
    return min(1.0, consonance / total_weight)


@njit(cache=True, fastmath=True)
def roughness_model(chroma: np.ndarray) -> float:
    """
    Compute roughness based on Plomp-Levelt (1965) beating model.
    
    Adjacent semitones and tritones create the most roughness.
    """
    roughness = 0.0
    
    # Minor second (1 semitone) - maximum roughness
    for i in range(11):
        r = chroma[i] * chroma[i + 1] * 0.8
        roughness += r
    # Wrap around
    roughness += chroma[11] * chroma[0] * 0.8
    
    # Major second (2 semitones) - still rough
    for i in range(10):
        roughness += chroma[i] * chroma[i + 2] * 0.4
    roughness += chroma[10] * chroma[0] * 0.4
    roughness += chroma[11] * chroma[1] * 0.4
    
    # Tritone (6 semitones) - dissonant
    for i in range(6):
        roughness += chroma[i] * chroma[i + 6] * 0.3
    
    return min(1.0, roughness)


@njit(cache=True, fastmath=True)
def harmonic_tension_model(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """
    Compute harmonic tension between two chroma vectors.
    
    Based on circle of fifths distance and interval relationships.
    """
    # Find dominant pitch classes
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
    
    # Interval between dominant pitches
    interval = abs(max1_idx - max2_idx)
    if interval > 6:
        interval = 12 - interval
    
    # Tension by interval:
    # 0: unison (0), 1: m2 (0.9), 2: M2 (0.6), 3: m3 (0.3), 
    # 4: M3 (0.25), 5: P4 (0.1), 6: tritone (0.85)
    tension_map = np.array([0.0, 0.9, 0.6, 0.3, 0.25, 0.1, 0.85])
    base_tension = tension_map[interval]
    
    # Modulate by chroma similarity (similar chromas = less tension)
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


def estimate_harmonicity(mfcc: np.ndarray) -> float:
    """
    Estimate harmonicity from MFCC.
    
    Harmonic sounds have smoother spectral envelopes.
    """
    if len(mfcc) < 3:
        return 0.5
    
    # Higher MFCCs represent fine spectral detail
    # Harmonic sounds have lower high-order MFCCs
    high_order = mfcc[10:] if len(mfcc) > 10 else mfcc[len(mfcc)//2:]
    low_order = mfcc[1:5] if len(mfcc) > 5 else mfcc[:len(mfcc)//2]
    
    high_energy = np.sum(high_order ** 2)
    low_energy = np.sum(low_order ** 2) + 1e-10
    
    # Low ratio of high/low = more harmonic
    ratio = high_energy / low_energy
    harmonicity = 1.0 / (1.0 + ratio * 2)
    
    return harmonicity


def compute_gestalt_continuity(ctx: TransitionContext) -> float:
    """
    Compute Gestalt continuity score.
    
    Based on auditory Gestalt principles:
    - Good continuation: similar features continue
    - Proximity: nearby features group together
    - Similarity: similar sounds are perceived as continuous
    """
    # Spectral similarity (good continuation)
    spectral_sim = (
        1.0 - abs(ctx.centroid_start - ctx.centroid_end) * 3 +
        1.0 - abs(ctx.bandwidth_start - ctx.bandwidth_end) * 3
    ) / 2
    spectral_sim = max(0, min(1, spectral_sim))
    
    # Loudness proximity
    loudness_prox = 1.0 - abs(ctx.loudness_start - ctx.loudness_end) * 4
    loudness_prox = max(0, min(1, loudness_prox))
    
    # MFCC similarity (overall timbral similarity)
    mfcc_sim = cosine_similarity(ctx.mfcc_start, ctx.mfcc_end)
    
    return (spectral_sim * 0.3 + loudness_prox * 0.3 + mfcc_sim * 0.4)


def score_tension_resolution(ctx: TransitionContext) -> float:
    """
    Score whether the transition respects tension-resolution patterns.
    
    Cutting during high tension (expecting resolution) sounds jarring.
    """
    # High transients suggest tension/attack
    tension_level = (ctx.transient_start + ctx.transient_end) / 2
    
    # High spectral flux suggests change/tension
    flux_tension = (ctx.flux_start + ctx.flux_end) / 2
    
    combined_tension = tension_level * 0.6 + flux_tension * 0.4
    
    # Low tension at cut point = good
    return 1.0 - combined_tension


class TensionProfiler:
    """
    Analyze musical tension profile for intelligent loop placement.
    
    Identifies points of high/low tension to avoid cutting during
    unresolved musical phrases.
    """
    
    def __init__(self):
        self.tension_curve = None
        self.resolution_points = None
    
    def analyze(self, features):
        """Build tension profile from audio features."""
        n_frames = features.n_frames
        
        # Compute frame-by-frame tension
        self.tension_curve = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(n_frames):
            chroma = features.chroma_cens[:, i]
            
            # Harmonic tension (dissonance)
            roughness = roughness_model(chroma)
            
            # Dynamic tension (loud = tense)
            loudness = features.loudness[i]
            
            # Spectral tension (bright = tense)
            brightness = features.spectral_centroid[i]
            
            # Transient tension
            transient = features.transient_strength[i]
            
            self.tension_curve[i] = (
                roughness * 0.3 +
                loudness * 0.25 +
                brightness * 0.2 +
                transient * 0.25
            )
        
        # Find resolution points (local minima in tension)
        from scipy.signal import find_peaks
        neg_tension = -self.tension_curve
        peaks, _ = find_peaks(neg_tension, distance=features.beat_length)
        self.resolution_points = peaks
        
        return self
    
    def score_loop_point(self, frame: int, tolerance: int = 5) -> float:
        """Score how good a frame is for loop placement."""
        if self.tension_curve is None:
            return 0.5
        
        # Low tension is good
        tension = self.tension_curve[frame]
        tension_score = 1.0 - tension
        
        # Near resolution point is good
        if self.resolution_points is not None and len(self.resolution_points) > 0:
            distances = np.abs(self.resolution_points - frame)
            min_dist = np.min(distances)
            resolution_score = max(0, 1.0 - min_dist / tolerance)
        else:
            resolution_score = 0.5
        
        return tension_score * 0.7 + resolution_score * 0.3

