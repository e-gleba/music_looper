"""
Audio Transitions Module - Smooth Crossfades and Fade Effects.

Provides high-quality audio transition effects for seamless looping:
- Crossfade transitions
- Fade in/out effects
- Adaptive fade length selection
- Spectral smoothing
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import butter, filtfilt


# ============================================================================
# CROSSFADE FUNCTIONS
# ============================================================================

def linear_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    fade_length: int,
) -> np.ndarray:
    """
    Perform linear crossfade between two audio segments.
    
    Args:
        audio1: First audio segment (fade out)
        audio2: Second audio segment (fade in)
        fade_length: Length of crossfade in samples
    
    Returns:
        Crossfaded audio segment
    """
    if fade_length <= 0:
        return np.concatenate((audio1, audio2))
    
    fade_length = min(fade_length, len(audio1), len(audio2))
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, fade_length)
    fade_in = np.linspace(0.0, 1.0, fade_length)
    
    # Apply crossfade
    if len(audio1.shape) == 1:
        # Mono
        result = np.zeros(len(audio1) + len(audio2) - fade_length, dtype=audio1.dtype)
        result[:len(audio1) - fade_length] = audio1[:-fade_length]
        result[len(audio1) - fade_length:len(audio1)] = (
            audio1[-fade_length:] * fade_out + audio2[:fade_length] * fade_in
        )
        result[len(audio1):] = audio2[fade_length:]
    else:
        # Multi-channel - vectorized across all channels
        result = np.zeros(
            (len(audio1) + len(audio2) - fade_length, audio1.shape[1]),
            dtype=audio1.dtype
        )
        result[:len(audio1) - fade_length] = audio1[:-fade_length]
        # Vectorized crossfade for all channels at once
        fade_out_2d = fade_out[:, np.newaxis]  # Shape: (fade_length, 1)
        fade_in_2d = fade_in[:, np.newaxis]    # Shape: (fade_length, 1)
        result[len(audio1) - fade_length:len(audio1)] = (
            audio1[-fade_length:] * fade_out_2d + audio2[:fade_length] * fade_in_2d
        )
        result[len(audio1):] = audio2[fade_length:]
    
    return result


def cosine_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    fade_length: int,
) -> np.ndarray:
    """
    Perform cosine (smooth) crossfade between two audio segments.
    
    Cosine crossfade is smoother than linear and reduces audible artifacts.
    """
    if fade_length <= 0:
        return np.concatenate((audio1, audio2))
    
    fade_length = min(fade_length, len(audio1), len(audio2))
    
    # Create cosine fade curves (smoother than linear)
    fade_out = (1.0 + np.cos(np.linspace(0, np.pi, fade_length))) / 2.0
    fade_in = (1.0 - np.cos(np.linspace(0, np.pi, fade_length))) / 2.0
    
    # Apply crossfade
    if len(audio1.shape) == 1:
        # Mono
        result = np.zeros(len(audio1) + len(audio2) - fade_length, dtype=audio1.dtype)
        result[:len(audio1) - fade_length] = audio1[:-fade_length]
        result[len(audio1) - fade_length:len(audio1)] = (
            audio1[-fade_length:] * fade_out + audio2[:fade_length] * fade_in
        )
        result[len(audio1):] = audio2[fade_length:]
    else:
        # Multi-channel - vectorized across all channels
        result = np.zeros(
            (len(audio1) + len(audio2) - fade_length, audio1.shape[1]),
            dtype=audio1.dtype
        )
        result[:len(audio1) - fade_length] = audio1[:-fade_length]
        # Vectorized crossfade for all channels at once
        fade_out_2d = fade_out[:, np.newaxis]  # Shape: (fade_length, 1)
        fade_in_2d = fade_in[:, np.newaxis]    # Shape: (fade_length, 1)
        result[len(audio1) - fade_length:len(audio1)] = (
            audio1[-fade_length:] * fade_out_2d + audio2[:fade_length] * fade_in_2d
        )
        result[len(audio1):] = audio2[fade_length:]
    
    return result


def exponential_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    fade_length: int,
    curve: float = 2.0,
) -> np.ndarray:
    """
    Perform exponential crossfade (more aggressive fade).
    
    Args:
        curve: Exponential curve factor (>1 = faster fade)
    """
    if fade_length <= 0:
        return np.concatenate((audio1, audio2))
    
    fade_length = min(fade_length, len(audio1), len(audio2))
    
    # Create exponential fade curves
    x = np.linspace(0, 1, fade_length)
    fade_out = np.power(1.0 - x, curve)
    fade_in = np.power(x, curve)
    
    # Apply crossfade
    if len(audio1.shape) == 1:
        # Mono
        result = np.zeros(len(audio1) + len(audio2) - fade_length, dtype=audio1.dtype)
        result[:len(audio1) - fade_length] = audio1[:-fade_length]
        result[len(audio1) - fade_length:len(audio1)] = (
            audio1[-fade_length:] * fade_out + audio2[:fade_length] * fade_in
        )
        result[len(audio1):] = audio2[fade_length:]
    else:
        # Multi-channel - vectorized across all channels
        result = np.zeros(
            (len(audio1) + len(audio2) - fade_length, audio1.shape[1]),
            dtype=audio1.dtype
        )
        result[:len(audio1) - fade_length] = audio1[:-fade_length]
        # Vectorized crossfade for all channels at once
        fade_out_2d = fade_out[:, np.newaxis]  # Shape: (fade_length, 1)
        fade_in_2d = fade_in[:, np.newaxis]    # Shape: (fade_length, 1)
        result[len(audio1) - fade_length:len(audio1)] = (
            audio1[-fade_length:] * fade_out_2d + audio2[:fade_length] * fade_in_2d
        )
        result[len(audio1):] = audio2[fade_length:]
    
    return result


def adaptive_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int,
    min_ms: int = 20,
    max_ms: int = 80,
) -> np.ndarray:
    """
    Perform adaptive crossfade with optimal length selection.
    
    Analyzes the audio to find the best crossfade length.
    """
    # Analyze energy at transition
    window_samples = int(sr * 0.05)  # 50ms window
    window_samples = min(window_samples, len(audio1), len(audio2))
    
    if window_samples < 32:
        # Too short, use minimal crossfade
        return cosine_crossfade(audio1, audio2, int(sr * min_ms / 1000))
    
    # Check energy levels
    energy1 = np.mean(audio1[-window_samples:] ** 2)
    energy2 = np.mean(audio2[:window_samples] ** 2)
    
    # Check for transients
    transient1 = np.max(np.abs(audio1[-window_samples:]))
    transient2 = np.max(np.abs(audio2[:window_samples]))
    
    # Adaptive fade length based on content
    if transient1 > 0.5 or transient2 > 0.5:
        # High transients - use longer crossfade
        fade_ms = max_ms
    elif abs(energy1 - energy2) > 0.1:
        # Energy mismatch - use medium crossfade
        fade_ms = (min_ms + max_ms) // 2
    else:
        # Similar energy - shorter crossfade is OK
        fade_ms = min_ms
    
    fade_samples = int(sr * fade_ms / 1000)
    
    # Use cosine crossfade for smoothness
    return cosine_crossfade(audio1, audio2, fade_samples)


# ============================================================================
# FADE IN/OUT EFFECTS
# ============================================================================

def fade_in(audio: np.ndarray, fade_length: int) -> np.ndarray:
    """Apply fade-in effect to audio."""
    if fade_length <= 0 or fade_length > len(audio):
        return audio
    
    result = audio.copy()
    fade_curve = np.linspace(0.0, 1.0, fade_length)
    
    if len(audio.shape) == 1:
        result[:fade_length] *= fade_curve
    else:
        # Vectorized: broadcast fade_curve across all channels
        result[:fade_length] *= fade_curve[:, np.newaxis]
    
    return result


def fade_out(audio: np.ndarray, fade_length: int) -> np.ndarray:
    """Apply fade-out effect to audio."""
    if fade_length <= 0 or fade_length > len(audio):
        return audio
    
    result = audio.copy()
    fade_curve = np.linspace(1.0, 0.0, fade_length)
    
    if len(audio.shape) == 1:
        result[-fade_length:] *= fade_curve
    else:
        # Vectorized: broadcast fade_curve across all channels
        result[-fade_length:] *= fade_curve[:, np.newaxis]
    
    return result


# ============================================================================
# SPECTRAL SMOOTHING
# ============================================================================

def spectral_smooth(audio: np.ndarray, sr: int, window_ms: int = 10) -> np.ndarray:
    """
    Apply spectral smoothing to reduce artifacts.
    
    Uses a gentle high-frequency filter to smooth transitions.
    """
    if len(audio.shape) == 2:
        # Multi-channel - vectorized using scipy.ndimage
        from scipy.ndimage import uniform_filter1d
        window_samples = max(1, int(sr * window_ms / 1000))
        return uniform_filter1d(audio, size=window_samples, axis=0, mode='constant')
    
    # Gentle low-pass filter to smooth high-frequency artifacts
    nyquist = sr / 2
    cutoff = min(nyquist * 0.95, 18000)  # Gentle filter, preserve most content
    
    try:
        b, a = butter(4, cutoff / nyquist, btype='low')
        return filtfilt(b, a, audio)
    except Exception:
        # If filtering fails, return original
        return audio


# ============================================================================
# TRANSITION QUALITY ANALYSIS
# ============================================================================

def analyze_transition_quality(
    audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    sr: int,
) -> dict:
    """
    Analyze transition quality and recommend crossfade parameters.
    
    Returns:
        dict with recommended fade_ms, fade_type, and quality_score
    """
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    
    pre_end = min(end_sample, len(audio))
    pre_start = max(0, pre_end - window_samples)
    post_start = max(0, start_sample)
    post_end = min(len(audio), post_start + window_samples)
    
    actual_len = min(pre_end - pre_start, post_end - post_start)
    if actual_len < 32:
        return {
            'fade_ms': 20,
            'fade_type': 'cosine',
            'quality_score': 0.5
        }
    
    pre = audio[pre_end - actual_len:pre_end]
    post = audio[post_start:post_start + actual_len]
    
    # Analyze energy
    energy_pre = np.mean(pre ** 2)
    energy_post = np.mean(post ** 2)
    energy_diff = abs(energy_pre - energy_post) / (max(energy_pre, energy_post) + 1e-10)
    
    # Analyze transients
    transient_pre = np.max(np.abs(pre))
    transient_post = np.max(np.abs(post))
    has_transients = max(transient_pre, transient_post) > 0.5
    
    # Analyze spectral similarity
    if actual_len >= 64:
        pre_fft = np.abs(np.fft.rfft(pre * np.hanning(len(pre))))
        post_fft = np.abs(np.fft.rfft(post * np.hanning(len(post))))
        pre_fft = pre_fft / (np.sum(pre_fft) + 1e-10)
        post_fft = post_fft / (np.sum(post_fft) + 1e-10)
        spectral_sim = np.dot(pre_fft, post_fft) / (
            np.linalg.norm(pre_fft) * np.linalg.norm(post_fft) + 1e-10
        )
    else:
        spectral_sim = 0.5
    
    # Analyze phase coherence (for better alignment)
    phase_coherence = 0.5
    if actual_len >= 128:
        try:
            from scipy.signal import correlate
            # Vectorized cross-correlation
            corr = correlate(pre, post, mode='valid')
            if len(corr) > 0:
                max_corr_idx = np.argmax(np.abs(corr))
                phase_coherence = np.abs(corr[max_corr_idx]) / (
                    np.linalg.norm(pre) * np.linalg.norm(post) + 1e-10
                )
        except Exception:
            pass
    
    # Analyze zero-crossing rate match (for smooth transitions)
    zcr_match = 0.5
    if len(pre) > 1 and len(post) > 1:
        # Vectorized zero-crossing detection
        pre_sign = np.sign(pre)
        post_sign = np.sign(post)
        pre_zcr = np.mean(np.abs(np.diff(pre_sign))) / 2.0
        post_zcr = np.mean(np.abs(np.diff(post_sign))) / 2.0
        zcr_match = 1.0 - abs(pre_zcr - post_zcr) / (max(pre_zcr, post_zcr) + 1e-10)
    
    # Detect vocal/ensemble content (for longer fades)
    has_vocals = False
    has_ensemble = False
    try:
        # Simple detection: check spectral characteristics
        # Vocals/ensemble typically have energy in 200-3000 Hz range
        if actual_len >= 64:
            pre_fft = np.abs(np.fft.rfft(pre * np.hanning(len(pre))))
            post_fft = np.abs(np.fft.rfft(post * np.hanning(len(post))))
            freqs = np.fft.fftfreq(len(pre_fft) * 2, 1/sr)[:len(pre_fft)]
            
            # Check vocal/ensemble range
            vocal_range = (freqs >= 200) & (freqs <= 3000)
            if np.any(vocal_range):
                pre_vocal_energy = np.sum(pre_fft[vocal_range])
                post_vocal_energy = np.sum(post_fft[vocal_range])
                pre_total = np.sum(pre_fft)
                post_total = np.sum(post_fft)
                
                pre_ratio = pre_vocal_energy / (pre_total + 1e-10)
                post_ratio = post_vocal_energy / (post_total + 1e-10)
                
                # High ratio suggests vocals/ensemble
                has_vocals = pre_ratio > 0.3 or post_ratio > 0.3
                has_ensemble = pre_ratio > 0.25 or post_ratio > 0.25
    except Exception:
        pass
    
    # Enhanced fade parameter recommendation
    # Combine multiple factors for better decision
    if has_transients:
        fade_ms = 70  # Longer fade for transients (drums, attacks)
        fade_type = 'exponential'  # More aggressive fade
    elif has_vocals or has_ensemble:
        # Vocals/ensemble need longer fades to hide breaks
        fade_ms = 80 if has_vocals else 60  # Even longer for vocals
        fade_type = 'cosine'
    elif energy_diff > 0.4:
        fade_ms = 50  # Medium-long fade for large energy mismatch
        fade_type = 'cosine'
    elif energy_diff > 0.2:
        fade_ms = 35  # Medium fade for moderate energy difference
        fade_type = 'cosine'
    elif spectral_sim < 0.6:
        fade_ms = 40  # Medium fade for spectral mismatch
        fade_type = 'cosine'
    else:
        fade_ms = 25  # Shorter fade for good match
        fade_type = 'cosine'
    
    # Adjust based on phase coherence
    if phase_coherence < 0.5:
        fade_ms = int(fade_ms * 1.3)  # Increase fade if poor phase alignment
    
    # Additional adjustment for vocals/ensemble
    if has_vocals:
        fade_ms = int(fade_ms * 1.2)  # 20% longer for vocals
    
    # Quality score combines multiple factors (vectorized)
    quality_score = (
        (1.0 - energy_diff) * 0.3 +
        spectral_sim * 0.3 +
        phase_coherence * 0.2 +
        zcr_match * 0.2
    )
    
    return {
        'fade_ms': fade_ms,
        'fade_type': fade_type,
        'quality_score': quality_score,
        'has_transients': has_transients,
        'energy_diff': float(energy_diff),
        'spectral_sim': float(spectral_sim),
        'phase_coherence': float(phase_coherence),
        'zcr_match': float(zcr_match)
    }

