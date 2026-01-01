"""
Psychoacoustic Expert - Human Auditory Perception Modeling.

Models human perception based on:
- Critical band theory (Bark scale)
- Temporal masking (forward/backward)
- Loudness perception (Fletcher-Munson curves, A-weighting)
- Just Noticeable Differences (JND)
- Auditory stream segregation
- Spectral masking curves
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


# Bark scale critical band edges (Hz)
BARK_EDGES = np.array([
    20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
    1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
    5300, 6400, 7700, 9500, 12000, 15500, 20000
], dtype=np.float32)

# ERB (Equivalent Rectangular Bandwidth) constants
ERB_CONST = 24.7
ERB_FACTOR = 9.265

# JND thresholds (Just Noticeable Differences)
JND_LOUDNESS_DB = 0.5    # ~0.5 dB for loudness
JND_PITCH_CENTS = 5.0    # ~5 cents for pitch
JND_TIMING_MS = 10.0     # ~10ms for timing

# Forward masking time constant (ms)
FORWARD_MASKING_MS = 200
BACKWARD_MASKING_MS = 50


class PsychoacousticExpert(Expert):
    """
    Expert modeling human auditory perception for loop quality.
    
    Uses psychoacoustic principles to predict whether a transition
    will be perceptible to human listeners.
    """
    
    name = "psychoacoustic"
    weight = 0.16
    
    def score(self, ctx: TransitionContext) -> float:
        """
        Score based on psychoacoustic imperceptibility.
        
        Higher score = transition is less likely to be perceived.
        """
        
        # 1. Masking potential at transition
        # High masking = transitions are less audible
        masking_score = (ctx.masking_start + ctx.masking_end) / 2
        
        # 2. Loudness continuity (JND-based)
        loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
        # Assuming loudness is normalized 0-1, map to perceptual threshold
        jnd_threshold = 0.03  # ~0.5dB in normalized scale
        if loudness_diff < jnd_threshold:
            loudness_score = 1.0
        elif loudness_diff < jnd_threshold * 3:
            loudness_score = 0.7
        else:
            loudness_score = max(0, 1.0 - loudness_diff * 3)
        
        # 3. RMS energy continuity
        rms_diff = abs(ctx.rms_start - ctx.rms_end)
        rms_score = max(0, 1.0 - rms_diff * 4)
        
        # 4. Spectral masking curve similarity
        # Low flux at transition = spectral content is stable = masking is effective
        spectral_stability = 1.0 - (ctx.flux_start + ctx.flux_end) / 2
        
        # 5. Temporal masking effectiveness
        # Transients provide forward masking for subsequent content
        # High transients BEFORE the loop point help mask the transition
        temporal_masking = compute_temporal_masking_score(
            ctx.transient_start, ctx.transient_end,
            ctx.onset_start, ctx.onset_end
        )
        
        # 6. Perceptual continuity (based on MFCC - captures spectral envelope)
        from pymusiclooper.analysis.experts.base import cosine_similarity
        perceptual_continuity = cosine_similarity(ctx.mfcc_start, ctx.mfcc_end)
        
        # 7. Brightness continuity (spectral centroid in perceptual scale)
        brightness_diff = abs(ctx.centroid_start - ctx.centroid_end)
        brightness_score = max(0, 1.0 - brightness_diff * 4)
        
        # 8. Flatness similarity (noise vs tonal perception)
        flatness_match = 1.0 - abs(ctx.flatness_start - ctx.flatness_end)
        
        # Combine with perceptual weights
        score = (
            masking_score * 0.18 +
            loudness_score * 0.15 +
            rms_score * 0.12 +
            spectral_stability * 0.15 +
            temporal_masking * 0.12 +
            perceptual_continuity * 0.13 +
            brightness_score * 0.08 +
            flatness_match * 0.07
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        masking = (ctx.masking_start + ctx.masking_end) / 2
        loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
        
        return (
            f"Psychoacoustic: {score:.2f} | "
            f"Masking: {masking:.2f} | "
            f"Loudness diff: {loudness_diff:.3f} | "
            f"Flux: {(ctx.flux_start + ctx.flux_end)/2:.2f}"
        )


@njit(cache=True, fastmath=True)
def compute_temporal_masking_score(
    trans_start: float, trans_end: float,
    onset_start: float, onset_end: float
) -> float:
    """
    Compute temporal masking effectiveness at transition.
    
    Forward masking: A loud sound masks quieter sounds that follow
    Backward masking: Less strong, but high-energy sounds after can help mask
    
    For loops: We want LOW transients at the cut point itself,
    but moderate transients nearby can help with masking.
    """
    # Transients AT the cut point are bad (audible discontinuity)
    trans_at_cut = max(trans_start, trans_end)
    
    if trans_at_cut > 0.7:
        # Very high transient at cut = very audible
        return 0.1
    elif trans_at_cut > 0.5:
        return 0.3
    elif trans_at_cut > 0.3:
        return 0.6
    else:
        # Low transient = good for masking
        return 0.8 + (0.3 - trans_at_cut) * 0.6


@njit(cache=True, fastmath=True)
def hz_to_bark(hz: float) -> float:
    """Convert frequency (Hz) to Bark scale."""
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


@njit(cache=True, fastmath=True)
def hz_to_erb(hz: float) -> float:
    """Convert frequency (Hz) to ERB (Equivalent Rectangular Bandwidth) scale."""
    return 24.7 * (4.37 * hz / 1000 + 1)


@njit(cache=True, fastmath=True)
def compute_spreading_function(bark_diff: float) -> float:
    """
    Compute the spreading function for spectral masking.
    
    Based on Schroeder spreading function - how much a masker
    at one Bark frequency masks sounds at other frequencies.
    """
    if bark_diff < 0:
        # Lower slope (masking of lower frequencies)
        return 10 ** (2.7 * bark_diff)
    else:
        # Upper slope (masking of higher frequencies)
        return 10 ** (-1.0 * bark_diff)


def compute_masking_threshold(
    power_spectrum: np.ndarray,
    sr: int,
    n_fft: int = 2048
) -> np.ndarray:
    """
    Compute psychoacoustic masking threshold curve.
    
    Implements a simplified version of the ISO MPEG psychoacoustic model.
    Returns the threshold below which sounds are masked.
    """
    n_bins = len(power_spectrum)
    threshold = np.zeros(n_bins, dtype=np.float32)
    
    # Frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1/sr)[:n_bins]
    
    # Convert to Bark scale
    bark_freqs = np.array([hz_to_bark(f) for f in freqs])
    
    # For each frequency bin, compute masking from nearby bins
    for i in range(n_bins):
        if freqs[i] < 20:  # Below hearing threshold
            continue
            
        masking_sum = 0.0
        for j in range(n_bins):
            if power_spectrum[j] > 1e-10:
                bark_diff = bark_freqs[i] - bark_freqs[j]
                spread = compute_spreading_function(bark_diff)
                masking_sum += power_spectrum[j] * spread
        
        # Threshold is sum of spreading minus absolute threshold
        threshold[i] = masking_sum
    
    return threshold


def compute_absolute_threshold(freq: float) -> float:
    """
    Compute absolute threshold of hearing at given frequency.
    
    Based on ISO 226 equal-loudness contours.
    Returns threshold in dB SPL.
    """
    if freq < 20 or freq > 20000:
        return 100.0  # Outside audible range
    
    # Simplified formula based on Fletcher-Munson curves
    f2 = (freq / 1000) ** 2
    ath = (
        3.64 * (freq / 1000) ** -0.8 -
        6.5 * np.exp(-0.6 * (freq / 1000 - 3.3) ** 2) +
        1e-3 * f2 * f2
    )
    
    return max(0, ath)


class CriticalBandAnalyzer:
    """Analyze audio in psychoacoustic critical bands."""
    
    def __init__(self, sr: int = 44100, n_bands: int = 24):
        self.sr = sr
        self.n_bands = n_bands
        self.bark_edges = BARK_EDGES[:n_bands + 1]
    
    def compute_band_energies(self, spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Compute energy in each critical band."""
        energies = np.zeros(self.n_bands, dtype=np.float32)
        
        for i in range(self.n_bands):
            low = self.bark_edges[i]
            high = self.bark_edges[i + 1]
            
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                energies[i] = np.sum(spectrum[mask] ** 2)
        
        return energies
    
    def compute_band_similarity(
        self, 
        spectrum1: np.ndarray, 
        spectrum2: np.ndarray,
        freqs: np.ndarray
    ) -> float:
        """Compute perceptual similarity based on critical band energies."""
        e1 = self.compute_band_energies(spectrum1, freqs)
        e2 = self.compute_band_energies(spectrum2, freqs)
        
        # Log-compress for perceptual relevance
        e1_db = 10 * np.log10(e1 + 1e-10)
        e2_db = 10 * np.log10(e2 + 1e-10)
        
        # Weighted difference (higher bands less important)
        weights = np.linspace(1.0, 0.5, self.n_bands)
        weighted_diff = np.abs(e1_db - e2_db) * weights
        
        # Normalize to similarity score
        max_diff = 60  # 60 dB max difference
        similarity = 1.0 - np.mean(weighted_diff) / max_diff
        
        return max(0, min(1, similarity))

