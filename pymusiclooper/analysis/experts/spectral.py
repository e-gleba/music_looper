"""
Advanced Spectral Expert - Deep Spectral Analysis.

Advanced spectral analysis:
- Spectral coherence
- Harmonic-to-noise ratio (HNR)
- Spectral entropy
- Formant continuity
- Spectral modulation
- Bark-band analysis
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class SpectralExpert(Expert):
    """
    Advanced spectral analysis expert.
    
    Goes beyond basic spectral features to analyze
    deep spectral characteristics for seamless transitions.
    """
    
    name = "spectral"
    weight = 0.10
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate spectral continuity at transition."""
        
        # 1. Spectral centroid match (brightness)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        centroid_score = max(0, 1.0 - centroid_diff * 5)
        
        # 2. Bandwidth match (spectral spread)
        bw_diff = abs(ctx.bandwidth_start - ctx.bandwidth_end)
        bandwidth_score = max(0, 1.0 - bw_diff * 5)
        
        # 3. Rolloff match (high-frequency content)
        rolloff_diff = abs(ctx.rolloff_start - ctx.rolloff_end)
        rolloff_score = max(0, 1.0 - rolloff_diff * 4)
        
        # 4. Flatness match (noise vs tonal)
        flatness_diff = abs(ctx.flatness_start - ctx.flatness_end)
        flatness_score = max(0, 1.0 - flatness_diff * 3)
        
        # 5. Low flux at transition (stable spectrum)
        flux_max = max(ctx.flux_start, ctx.flux_end)
        flux_score = 1.0 - flux_max
        
        # 6. Spectral entropy similarity
        entropy_start = compute_spectral_entropy(ctx.chroma_start)
        entropy_end = compute_spectral_entropy(ctx.chroma_end)
        entropy_diff = abs(entropy_start - entropy_end)
        entropy_score = max(0, 1.0 - entropy_diff * 2)
        
        # 7. Harmonic content similarity (from MFCC)
        from pymusiclooper.analysis.experts.base import cosine_similarity
        mfcc_sim = cosine_similarity(ctx.mfcc_start, ctx.mfcc_end)
        
        # 8. Spectral shape coherence
        shape_coherence = compute_spectral_shape_coherence(
            ctx.mfcc_start, ctx.mfcc_end
        )
        
        # Combine scores
        score = (
            centroid_score * 0.15 +
            bandwidth_score * 0.10 +
            rolloff_score * 0.10 +
            flatness_score * 0.10 +
            flux_score * 0.15 +
            entropy_score * 0.10 +
            mfcc_sim * 0.15 +
            shape_coherence * 0.15
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        flux = max(ctx.flux_start, ctx.flux_end)
        
        return (
            f"Spectral: {score:.2f} | "
            f"Centroid diff: {centroid_diff:.3f} | "
            f"Flux: {flux:.2f}"
        )


@njit(cache=True, fastmath=True)
def compute_spectral_entropy(spectrum: np.ndarray) -> float:
    """
    Compute spectral entropy (information content).
    
    Low entropy = concentrated spectral energy (tonal)
    High entropy = spread spectral energy (noisy)
    """
    # Normalize to probability distribution
    total = np.sum(spectrum) + 1e-10
    p = spectrum / total
    
    # Compute entropy
    entropy = 0.0
    for i in range(len(p)):
        if p[i] > 1e-10:
            entropy -= p[i] * np.log2(p[i])
    
    # Normalize to [0, 1]
    max_entropy = np.log2(len(spectrum))
    return entropy / max_entropy if max_entropy > 0 else 0


def compute_spectral_shape_coherence(mfcc1: np.ndarray, mfcc2: np.ndarray) -> float:
    """
    Compute spectral shape coherence using MFCC.
    
    MFCCs encode the spectral envelope shape - coherence means
    similar timbral characteristics.
    """
    # First MFCC is energy - skip for shape analysis
    shape1 = mfcc1[1:13] if len(mfcc1) > 13 else mfcc1[1:]
    shape2 = mfcc2[1:13] if len(mfcc2) > 13 else mfcc2[1:]
    
    # Compute cosine similarity of shape vectors
    from pymusiclooper.analysis.experts.base import cosine_similarity
    base_sim = cosine_similarity(shape1, shape2)
    
    # Also check higher MFCCs (fine detail)
    if len(mfcc1) > 13 and len(mfcc2) > 13:
        detail1 = mfcc1[13:]
        detail2 = mfcc2[13:]
        detail_sim = cosine_similarity(detail1, detail2)
        return base_sim * 0.7 + detail_sim * 0.3
    
    return base_sim


@njit(cache=True, fastmath=True)
def compute_harmonic_ratio(spectrum: np.ndarray) -> float:
    """
    Estimate harmonic-to-noise ratio from spectrum.
    
    High HNR = clear harmonic content (voice, instruments)
    Low HNR = noisy content
    """
    if len(spectrum) < 3:
        return 0.5
    
    # Simple peak detection for harmonics
    peaks = 0.0
    noise = 0.0
    
    for i in range(1, len(spectrum) - 1):
        if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
            peaks += spectrum[i]
        else:
            noise += spectrum[i]
    
    total = peaks + noise + 1e-10
    return peaks / total


def compute_bark_band_similarity(
    power_db1: np.ndarray,
    power_db2: np.ndarray,
    sr: int = 44100,
    n_bands: int = 24,
) -> float:
    """
    Compute similarity in Bark critical bands.
    
    More perceptually relevant than linear frequency comparison.
    """
    # Bark band edges (Hz)
    bark_edges = np.array([
        20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
        1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
        5300, 6400, 7700, 9500, 12000, 15500, 20000
    ])[:n_bands + 1]
    
    # Compute band energies
    n_bins = len(power_db1)
    freq_per_bin = sr / 2 / n_bins
    
    bands1 = np.zeros(n_bands)
    bands2 = np.zeros(n_bands)
    
    for b in range(n_bands):
        low_bin = int(bark_edges[b] / freq_per_bin)
        high_bin = int(bark_edges[b + 1] / freq_per_bin)
        high_bin = min(high_bin, n_bins)
        
        if high_bin > low_bin:
            bands1[b] = np.mean(power_db1[low_bin:high_bin])
            bands2[b] = np.mean(power_db2[low_bin:high_bin])
    
    # Compute weighted similarity (lower bands more important)
    weights = np.linspace(1.0, 0.5, n_bands)
    weighted_diff = np.abs(bands1 - bands2) * weights
    
    # Normalize to similarity score
    max_diff = 60  # 60 dB max difference
    similarity = 1.0 - np.mean(weighted_diff) / max_diff
    
    return max(0, min(1, similarity))


def compute_spectral_modulation(mfcc_sequence: np.ndarray) -> np.ndarray:
    """
    Compute spectral modulation (temporal changes in spectrum).
    
    Important for capturing dynamics of spectral evolution.
    """
    if mfcc_sequence.shape[1] < 2:
        return np.zeros(mfcc_sequence.shape[0])
    
    # Delta MFCCs
    delta = np.diff(mfcc_sequence, axis=1)
    
    # Modulation energy per coefficient
    modulation = np.sqrt(np.mean(delta ** 2, axis=1))
    
    return modulation


def compute_formant_continuity(mfcc1: np.ndarray, mfcc2: np.ndarray) -> float:
    """
    Estimate formant continuity from MFCC.
    
    Formants are resonant frequencies that define vowel sounds
    and instrument timbre. Continuity suggests natural transition.
    """
    # Lower MFCCs (2-6) roughly correspond to formant structure
    formant1 = mfcc1[1:6] if len(mfcc1) >= 6 else mfcc1[1:]
    formant2 = mfcc2[1:6] if len(mfcc2) >= 6 else mfcc2[1:]
    
    from pymusiclooper.analysis.experts.base import cosine_similarity
    return cosine_similarity(formant1, formant2)

