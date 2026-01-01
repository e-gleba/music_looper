"""
Crossfade Expert - Waveform-Level Transition Quality.

Detailed waveform analysis for crossfade quality:
- Waveform correlation
- Energy envelope matching
- Zero-crossing compatibility
- Phase alignment
- Transient-free regions
- Spectral phase coherence
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, hilbert

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class CrossfadeExpert(Expert):
    """
    Expert for sample-level crossfade quality.
    
    Analyzes raw waveform characteristics to predict
    how audible a crossfade will be.
    """
    
    name = "crossfade"
    weight = 0.12
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate waveform compatibility for crossfade."""
        
        # Get audio regions around transition
        fade_ms = 50
        fade_samples = int(ctx.sr * fade_ms / 1000)
        
        start_samples = ctx.start_frame * ctx.hop
        end_samples = ctx.end_frame * ctx.hop
        
        audio_len = len(ctx.audio)
        
        # Pre-transition region (end of loop)
        pre_end = min(end_samples, audio_len)
        pre_start = max(0, pre_end - fade_samples)
        
        # Post-transition region (start of loop)
        post_start = max(0, start_samples)
        post_end = min(audio_len, post_start + fade_samples)
        
        fade_len = min(pre_end - pre_start, post_end - post_start)
        
        if fade_len < 32:
            return 0.3  # Too short for reliable analysis
        
        pre = ctx.audio[pre_end - fade_len:pre_end]
        post = ctx.audio[post_start:post_start + fade_len]
        
        # 1. Waveform correlation
        corr_score = compute_waveform_correlation(pre, post)
        
        # 2. Energy envelope match
        env_score = compute_envelope_match(pre, post)
        
        # 3. Zero-crossing rate match
        zcr_score = compute_zcr_match(pre, post)
        
        # 4. Low amplitude at cut points
        cut_amp_score = score_cut_amplitude(pre, post)
        
        # 5. Phase alignment
        phase_score = compute_phase_alignment(pre, post)
        
        # 6. Transient absence at cut
        transient_score = score_transient_absence(pre, post)
        
        # 7. Simulated crossfade smoothness
        smoothness_score = compute_crossfade_smoothness(pre, post)
        
        # 8. Spectral similarity in fade region
        spectral_score = compute_fade_spectral_match(pre, post)
        
        # Combine scores
        score = (
            corr_score * 0.15 +
            env_score * 0.15 +
            zcr_score * 0.10 +
            cut_amp_score * 0.10 +
            phase_score * 0.15 +
            transient_score * 0.15 +
            smoothness_score * 0.10 +
            spectral_score * 0.10
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        return f"Crossfade: {score:.2f}"


def compute_waveform_correlation(pre: np.ndarray, post: np.ndarray) -> float:
    """Compute waveform correlation at transition."""
    std_pre = np.std(pre)
    std_post = np.std(post)
    
    if std_pre < 1e-8 or std_post < 1e-8:
        return 0.3
    
    corr = np.corrcoef(pre, post)[0, 1]
    return max(0, corr) if not np.isnan(corr) else 0.3


def compute_envelope_match(pre: np.ndarray, post: np.ndarray) -> float:
    """Compare energy envelopes of fade regions."""
    n_windows = min(8, len(pre) // 64)
    if n_windows < 2:
        e_pre = np.mean(pre ** 2)
        e_post = np.mean(post ** 2)
        e_diff = abs(e_pre - e_post) / (max(e_pre, e_post) + 1e-10)
        return 1.0 - min(e_diff, 1.0)
    
    win_size = len(pre) // n_windows
    pre_env = np.array([np.mean(pre[j*win_size:(j+1)*win_size]**2) for j in range(n_windows)])
    post_env = np.array([np.mean(post[j*win_size:(j+1)*win_size]**2) for j in range(n_windows)])
    
    if np.std(pre_env) < 1e-10 or np.std(post_env) < 1e-10:
        return 0.5
    
    corr = np.corrcoef(pre_env, post_env)[0, 1]
    return max(0, corr) if not np.isnan(corr) else 0.5


def compute_zcr_match(pre: np.ndarray, post: np.ndarray) -> float:
    """Compare zero-crossing rates (rhythm/texture indicator)."""
    zcr_pre = np.sum(np.abs(np.diff(np.sign(pre)))) / (2 * len(pre) + 1e-10)
    zcr_post = np.sum(np.abs(np.diff(np.sign(post)))) / (2 * len(post) + 1e-10)
    
    return 1.0 - min(abs(zcr_pre - zcr_post) * 3, 1.0)


def score_cut_amplitude(pre: np.ndarray, post: np.ndarray) -> float:
    """Score amplitude at cut point (low is better)."""
    max_pre = np.max(np.abs(pre)) + 1e-10
    max_post = np.max(np.abs(post)) + 1e-10
    
    cut_amp_pre = abs(pre[-1]) / max_pre
    cut_amp_post = abs(post[0]) / max_post
    
    return 1.0 - (cut_amp_pre + cut_amp_post) / 2


def compute_phase_alignment(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Compute phase alignment using Hilbert transform.
    
    Aligned phase means smoother transition.
    """
    try:
        # Get instantaneous phase
        analytic_pre = hilbert(pre[-min(256, len(pre)):])
        analytic_post = hilbert(post[:min(256, len(post))])
        
        phase_pre = np.angle(analytic_pre)
        phase_post = np.angle(analytic_post)
        
        # Phase difference at cut point
        phase_diff = abs(phase_pre[-1] - phase_post[0])
        
        # Normalize to [0, pi]
        if phase_diff > np.pi:
            phase_diff = 2 * np.pi - phase_diff
        
        # Convert to score (0 = perfectly aligned, pi = worst)
        return 1.0 - phase_diff / np.pi
    except Exception:
        return 0.5


def score_transient_absence(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Score based on absence of transients near cut point.
    
    Cutting at transients creates audible clicks.
    """
    # Check energy spike at cut point
    window = min(64, len(pre) // 4, len(post) // 4)
    if window < 4:
        return 0.5
    
    # Energy before cut
    pre_cut_energy = np.mean(pre[-window:] ** 2)
    pre_avg_energy = np.mean(pre ** 2) + 1e-10
    pre_ratio = pre_cut_energy / pre_avg_energy
    
    # Energy after cut
    post_cut_energy = np.mean(post[:window] ** 2)
    post_avg_energy = np.mean(post ** 2) + 1e-10
    post_ratio = post_cut_energy / post_avg_energy
    
    # High ratio = transient present = bad
    max_ratio = max(pre_ratio, post_ratio)
    
    if max_ratio > 2.0:
        return 0.2
    elif max_ratio > 1.5:
        return 0.5
    else:
        return 0.8 + (1.0 - max_ratio) * 0.2


def compute_crossfade_smoothness(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Simulate crossfade and measure smoothness.
    
    Actually performs the crossfade and measures discontinuities.
    """
    fade_len = min(256, len(pre), len(post))
    
    fade_out = np.linspace(1, 0, fade_len)
    fade_in = np.linspace(0, 1, fade_len)
    
    # Simulate crossfade
    mixed = pre[-fade_len:] * fade_out + post[:fade_len] * fade_in
    
    # Measure smoothness (low variance of derivative = smooth)
    diff = np.diff(mixed)
    smoothness = 1.0 / (1.0 + np.std(diff) * 20)
    
    return smoothness


def compute_fade_spectral_match(pre: np.ndarray, post: np.ndarray) -> float:
    """
    Compare spectral content in fade region.
    
    Similar spectra = less audible transition.
    """
    # Use short-time FFT
    n_fft = min(256, len(pre), len(post))
    
    if n_fft < 64:
        return 0.5
    
    pre_fft = np.abs(np.fft.rfft(pre[-n_fft:] * np.hanning(n_fft)))
    post_fft = np.abs(np.fft.rfft(post[:n_fft] * np.hanning(n_fft)))
    
    # Normalize
    pre_fft = pre_fft / (np.sum(pre_fft) + 1e-10)
    post_fft = post_fft / (np.sum(post_fft) + 1e-10)
    
    # Cosine similarity
    dot = np.dot(pre_fft, post_fft)
    norm = np.linalg.norm(pre_fft) * np.linalg.norm(post_fft) + 1e-10
    
    return dot / norm


def find_optimal_crossfade_length(
    audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    sr: int,
    min_ms: int = 10,
    max_ms: int = 100,
) -> int:
    """
    Find optimal crossfade length for this transition.
    
    Longer crossfades are smoother but can cause phase cancellation.
    """
    best_length = min_ms
    best_score = 0.0
    
    for fade_ms in range(min_ms, max_ms + 1, 10):
        fade_samples = int(sr * fade_ms / 1000)
        
        pre_end = min(end_sample, len(audio))
        pre_start = max(0, pre_end - fade_samples)
        post_start = max(0, start_sample)
        post_end = min(len(audio), post_start + fade_samples)
        
        actual_len = min(pre_end - pre_start, post_end - post_start)
        if actual_len < 32:
            continue
        
        pre = audio[pre_end - actual_len:pre_end]
        post = audio[post_start:post_start + actual_len]
        
        score = compute_crossfade_smoothness(pre, post)
        
        if score > best_score:
            best_score = score
            best_length = fade_ms
    
    return best_length

