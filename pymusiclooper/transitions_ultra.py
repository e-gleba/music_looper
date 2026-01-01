"""
Ultra-Perfect Transitions - Maximum Information Extraction and Gap Masking.

Uses comprehensive segment analysis to create perfect transitions:
- Deep analysis of entire segments (not just transition points)
- Aggressive gap detection and masking
- Advanced time-stretching for rhythm gaps
- Full-spectrum analysis
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, correlate, butter, filtfilt
from scipy.fft import rfft, irfft, fftfreq
from scipy.ndimage import uniform_filter1d

try:
    from pymusiclooper.transitions_advanced_analysis import (
        analyze_segment_comprehensive,
        SegmentAnalysis
    )
    HAS_ADVANCED_ANALYSIS = True
except Exception:
    HAS_ADVANCED_ANALYSIS = False


def ultra_perfect_transition(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int,
    min_ms: int = 30,
    max_ms: int = 150,
) -> np.ndarray:
    """
    Ultra-perfect transition with maximum information extraction.
    
    Analyzes ENTIRE segments (not just transition points) to:
    1. Detect all rhythm gaps
    2. Detect all energy gaps
    3. Match spectral envelopes
    4. Match instrument presence
    5. Apply aggressive time-stretching for gaps
    6. Create seamless transition
    """
    # Analyze both segments comprehensively
    if HAS_ADVANCED_ANALYSIS:
        analysis1 = analyze_segment_comprehensive(audio1, sr)
        analysis2 = analyze_segment_comprehensive(audio2, sr)
    else:
        # Fallback to simpler analysis
        analysis1 = _simple_segment_analysis(audio1, sr)
        analysis2 = _simple_segment_analysis(audio2, sr)
    
    # Determine optimal fade length based on analysis
    fade_ms = _determine_optimal_fade_length(
        analysis1, analysis2, audio1, audio2, sr, min_ms, max_ms
    )
    fade_samples = int(sr * fade_ms / 1000)
    fade_samples = min(fade_samples, len(audio1), len(audio2))
    
    if fade_samples < 16:
        return np.concatenate((audio1, audio2))
    
    # Extract transition regions with context
    context_samples = int(sr * 0.2)  # 200ms context
    pre_start = max(0, len(audio1) - fade_samples - context_samples)
    pre_end = len(audio1)
    post_start = 0
    post_end = min(len(audio2), fade_samples + context_samples)
    
    pre_context = audio1[pre_start:pre_end]
    post_context = audio2[post_start:post_end]
    
    # Detect rhythm gap between segments
    rhythm_gap = _detect_rhythm_gap_advanced(
        pre_context, post_context, sr, analysis1, analysis2
    )
    
    # Detect spectral mismatch
    spectral_mismatch = _detect_spectral_mismatch(
        pre_context, post_context, sr, analysis1, analysis2
    )
    
    # Apply corrections
    corrected_post = post_context.copy()
    
    # 1. Time-stretching for rhythm gaps (ULTRA-AGGRESSIVE)
    # Apply even for tiny gaps - better to over-correct than have audible breaks
    if abs(rhythm_gap) > 0.5:  # Even 0.5ms gap gets correction
        gap_samples = int(sr * abs(rhythm_gap) / 1000)
        # Add 20% extra stretching for safety margin
        gap_samples = int(gap_samples * 1.2)
        
        if rhythm_gap > 0:
            # Gap - stretch post aggressively
            corrected_post = _aggressive_time_stretch(corrected_post, gap_samples, sr)
        else:
            # Overlap - compress post
            corrected_post = _aggressive_time_compress(corrected_post, abs(gap_samples), sr)
    
    # 2. Spectral matching
    if spectral_mismatch > 0.3:
        corrected_post = _match_spectral_envelope(
            pre_context[-fade_samples:],
            corrected_post[:fade_samples],
            sr
        )
        # Reconstruct full post with corrected beginning
        if len(corrected_post) < len(post_context):
            corrected_post = np.concatenate((
                corrected_post,
                post_context[len(corrected_post):]
            ))
        elif len(corrected_post) > len(post_context):
            corrected_post = corrected_post[:len(post_context)]
    
    # 3. Energy matching
    energy_diff = np.abs(
        np.mean(pre_context[-fade_samples:] ** 2) -
        np.mean(corrected_post[:fade_samples] ** 2)
    )
    if energy_diff > 0.2:
        corrected_post = _match_energy_envelope(
            pre_context[-fade_samples:],
            corrected_post[:fade_samples],
            sr
        )
        if len(corrected_post) < len(post_context):
            corrected_post = np.concatenate((
                corrected_post,
                post_context[len(corrected_post):]
            ))
        elif len(corrected_post) > len(post_context):
            corrected_post = corrected_post[:len(post_context)]
    
    # 4. Apply advanced crossfade
    from pymusiclooper.transitions_advanced import organic_crossfade
    
    try:
        result = organic_crossfade(
            audio1[:-fade_samples],
            corrected_post,
            sr,
            min_ms=fade_ms,
            max_ms=fade_ms,
            use_phase_alignment=True,
            use_spectral_morph=True,
            use_perceptual=True,
            use_eq_match=True,
        )
    except Exception:
        # Fallback
        from pymusiclooper.transitions import cosine_crossfade
        result = cosine_crossfade(
            audio1[:-fade_samples],
            corrected_post[:fade_samples],
            fade_samples
        )
        result = np.concatenate((result, corrected_post[fade_samples:]))
    
    return result


def _simple_segment_analysis(audio: np.ndarray, sr: int) -> dict:
    """Simple segment analysis fallback."""
    # Basic rhythm detection
    window_size = min(4096, len(audio) // 4)
    if window_size < 512:
        window_size = len(audio)
    
    autocorr = correlate(audio[:window_size], audio[:window_size], mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    
    # Find tempo
    min_period = int(sr / 300)
    max_period = int(sr / 60)
    if max_period < len(autocorr):
        search_range = autocorr[min_period:max_period]
        if len(search_range) > 0:
            peak_idx = np.argmax(search_range) + min_period
            tempo = sr / peak_idx * 60 if peak_idx > 0 else 120.0
        else:
            tempo = 120.0
    else:
        tempo = 120.0
    
    # Energy analysis
    window_energy = int(sr * 0.05)
    energy = []
    for i in range(0, len(audio) - window_energy, window_energy // 2):
        energy.append(np.mean(audio[i:i+window_energy] ** 2))
    energy = np.array(energy) if energy else np.array([np.mean(audio ** 2)])
    
    # Spectral analysis
    n_fft = 2 ** int(np.ceil(np.log2(min(2048, len(audio)))))
    fft = rfft(audio[:min(2048, len(audio))] * np.hanning(min(2048, len(audio))), n=n_fft)
    magnitude = np.abs(fft)
    freqs = fftfreq(n_fft, 1/sr)[:len(fft)]
    
    # Instrument detection
    vocal_range = (freqs >= 200) & (freqs <= 2000)
    ensemble_range = (freqs >= 200) & (freqs <= 3000)
    
    has_vocals = False
    has_ensemble = False
    if np.any(vocal_range):
        vocal_energy = np.sum(magnitude[vocal_range])
        total_energy = np.sum(magnitude)
        has_vocals = (vocal_energy / (total_energy + 1e-10)) > 0.3
    
    if np.any(ensemble_range):
        ensemble_energy = np.sum(magnitude[ensemble_range])
        total_energy = np.sum(magnitude)
        has_ensemble = (ensemble_energy / (total_energy + 1e-10)) > 0.25
    
    return {
        'tempo': tempo,
        'energy': energy,
        'has_vocals': has_vocals,
        'has_ensemble': has_ensemble,
        'beats': np.array([]),
        'rhythm_gaps': np.array([]),
    }


def _determine_optimal_fade_length(
    analysis1: dict | SegmentAnalysis,
    analysis2: dict | SegmentAnalysis,
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int,
    min_ms: int,
    max_ms: int,
) -> int:
    """Determine optimal fade length based on comprehensive analysis."""
    # Base fade length
    fade_ms = 50
    
    # Increase for vocals/ensemble
    if isinstance(analysis1, dict):
        has_vocals1 = analysis1.get('has_vocals', False)
        has_ensemble1 = analysis1.get('has_ensemble', False)
    else:
        has_vocals1 = getattr(analysis1, 'has_vocals', False)
        has_ensemble1 = getattr(analysis1, 'has_ensemble', False)
    
    if isinstance(analysis2, dict):
        has_vocals2 = analysis2.get('has_vocals', False)
        has_ensemble2 = analysis2.get('has_ensemble', False)
    else:
        has_vocals2 = getattr(analysis2, 'has_vocals', False)
        has_ensemble2 = getattr(analysis2, 'has_ensemble', False)
    
    has_vocals = has_vocals1 or has_vocals2
    has_ensemble = has_ensemble1 or has_ensemble2
    
    if has_vocals:
        fade_ms = 100  # Longer for vocals
    elif has_ensemble:
        fade_ms = 80  # Longer for ensemble
    
    # Increase for rhythm gaps
    if isinstance(analysis1, dict):
        rhythm_gaps1 = analysis1.get('rhythm_gaps', np.array([]))
    else:
        rhythm_gaps1 = getattr(analysis1, 'rhythm_gaps', np.array([]))
    
    if isinstance(analysis2, dict):
        rhythm_gaps2 = analysis2.get('rhythm_gaps', np.array([]))
    else:
        rhythm_gaps2 = getattr(analysis2, 'rhythm_gaps', np.array([]))
    
    if len(rhythm_gaps1) > 0 or len(rhythm_gaps2) > 0:
        fade_ms = int(fade_ms * 1.3)  # 30% longer if gaps detected
    
    # Clamp
    fade_ms = max(min_ms, min(max_ms, fade_ms))
    
    return fade_ms


def _detect_rhythm_gap_advanced(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    analysis1: dict | SegmentAnalysis,
    analysis2: dict | SegmentAnalysis,
) -> float:
    """Advanced rhythm gap detection using full segment analysis."""
    # Method 1: Use beat intervals from analysis
    if isinstance(analysis1, dict):
        intervals1 = analysis1.get('beat_intervals', np.array([]))
    else:
        intervals1 = getattr(analysis1, 'beat_intervals', np.array([]))
    
    if isinstance(analysis2, dict):
        intervals2 = analysis2.get('beat_intervals', np.array([]))
    else:
        intervals2 = getattr(analysis2, 'beat_intervals', np.array([]))
    
    if len(intervals1) > 0 and len(intervals2) > 0:
        mean_interval1 = np.mean(intervals1)
        mean_interval2 = np.mean(intervals2)
        if mean_interval1 > 0 and mean_interval2 > 0:
            gap = mean_interval2 - mean_interval1
            gap_ms = gap / sr * 1000
            return float(gap_ms)
    
    # Method 2: Direct autocorrelation on transition region
    window_size = min(2048, len(pre), len(post))
    if window_size < 512:
        return 0.0
    
    pre_window = pre[-window_size:]
    post_window = post[:window_size]
    
    # Autocorrelation
    pre_autocorr = correlate(pre_window, pre_window, mode='full')
    pre_autocorr = pre_autocorr[len(pre_autocorr) // 2:]
    post_autocorr = correlate(post_window, post_window, mode='full')
    post_autocorr = post_autocorr[len(post_autocorr) // 2:]
    
    # Find periods
    min_period = int(sr / 300)
    max_period = int(sr / 60)
    
    if max_period < len(pre_autocorr) and max_period < len(post_autocorr):
        pre_range = pre_autocorr[min_period:max_period]
        post_range = post_autocorr[min_period:max_period]
        
        if len(pre_range) > 0 and len(post_range) > 0:
            pre_period = np.argmax(pre_range) + min_period
            post_period = np.argmax(post_range) + min_period
            
            if pre_period > 0 and post_period > 0:
                gap = post_period - pre_period
                gap_ms = gap / sr * 1000
                return float(gap_ms)
    
    return 0.0


def _detect_spectral_mismatch(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    analysis1: dict | SegmentAnalysis | None = None,
    analysis2: dict | SegmentAnalysis | None = None,
) -> float:
    """Detect spectral mismatch between segments."""
    window_size = min(2048, len(pre), len(post))
    if window_size < 64:
        return 0.5
    
    pre_window = pre[-window_size:] * np.hanning(window_size)
    post_window = post[:window_size] * np.hanning(window_size)
    
    n_fft = 2 ** int(np.ceil(np.log2(window_size)))
    pre_fft = rfft(pre_window, n=n_fft)
    post_fft = rfft(post_window, n=n_fft)
    
    pre_mag = np.abs(pre_fft)
    post_mag = np.abs(post_fft)
    
    # Normalize
    pre_mag = pre_mag / (np.sum(pre_mag) + 1e-10)
    post_mag = post_mag / (np.sum(post_mag) + 1e-10)
    
    # Cosine similarity
    similarity = np.dot(pre_mag, post_mag) / (
        np.linalg.norm(pre_mag) * np.linalg.norm(post_mag) + 1e-10
    )
    
    mismatch = 1.0 - similarity
    return float(mismatch)


def _aggressive_time_stretch(audio: np.ndarray, stretch_samples: int, sr: int) -> np.ndarray:
    """Aggressive time-stretching for gap masking."""
    if stretch_samples <= 0 or len(audio) < 64:
        return audio
    
    # Use overlap-add phase vocoder
    analysis_hop = 256
    stretch_ratio = 1.0 + stretch_samples / len(audio)
    synthesis_hop = int(analysis_hop * stretch_ratio)
    window_size = 1024
    
    n_fft = 2 ** int(np.ceil(np.log2(window_size)))
    target_len = len(audio) + stretch_samples
    
    output = np.zeros(target_len, dtype=audio.dtype)
    window_sum = np.zeros(target_len, dtype=np.float32)
    
    prev_phase = None
    
    for i in range(0, len(audio) - window_size, analysis_hop):
        window = audio[i:i+window_size] * np.hanning(window_size)
        
        fft = rfft(window, n=n_fft)
        mag = np.abs(fft)
        phase = np.angle(fft)
        
        if prev_phase is not None:
            phase_diff = phase - prev_phase
            phase_diff = np.angle(np.exp(1j * phase_diff))
            phase = prev_phase + phase_diff
        
        synth_pos = int(i * stretch_ratio)
        
        if synth_pos + window_size <= target_len:
            synthesis_fft = mag * np.exp(1j * phase)
            synthesis_window = irfft(synthesis_fft, n=n_fft)[:window_size]
            synthesis_window *= np.hanning(window_size)
            
            output[synth_pos:synth_pos+window_size] += synthesis_window
            window_sum[synth_pos:synth_pos+window_size] += np.hanning(window_size)
        
        prev_phase = phase
    
    window_sum = np.maximum(window_sum, 1e-10)
    output = output / window_sum
    
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * np.max(np.abs(audio))
    
    return output[:target_len]


def _aggressive_time_compress(audio: np.ndarray, compress_samples: int, sr: int) -> np.ndarray:
    """Aggressive time-compression."""
    if compress_samples <= 0 or compress_samples >= len(audio):
        return audio
    
    target_len = max(1, len(audio) - compress_samples)
    
    # Use resampling with anti-aliasing
    indices = np.linspace(0, len(audio) - 1, target_len)
    compressed = np.interp(indices, np.arange(len(audio)), audio)
    
    return compressed


def _match_spectral_envelope(pre: np.ndarray, post: np.ndarray, sr: int) -> np.ndarray:
    """Match spectral envelope of post to pre."""
    if len(pre) < 64 or len(post) < 64:
        return post
    
    n_fft = 2 ** int(np.ceil(np.log2(max(len(pre), len(post)))))
    
    pre_fft = rfft(pre * np.hanning(len(pre)), n=n_fft)
    post_fft = rfft(post * np.hanning(len(post)), n=n_fft)
    
    pre_mag = np.abs(pre_fft)
    post_mag = np.abs(post_fft)
    
    # Compute matching filter
    pre_mag_smooth = uniform_filter1d(pre_mag, size=5)
    post_mag_smooth = uniform_filter1d(post_mag, size=5)
    
    match_filter = pre_mag_smooth / (post_mag_smooth + 1e-10)
    match_filter = np.clip(match_filter, 0.3, 3.0)  # Limit extreme corrections
    
    # Apply filter
    post_fft_matched = post_fft * match_filter
    post_matched = irfft(post_fft_matched, n=n_fft)[:len(post)]
    
    # Blend with original
    result = post * 0.4 + post_matched * 0.6
    
    return result


def _match_energy_envelope(pre: np.ndarray, post: np.ndarray, sr: int) -> np.ndarray:
    """Match energy envelope of post to pre."""
    if len(pre) < 16 or len(post) < 16:
        return post
    
    pre_energy = np.mean(pre ** 2)
    post_energy = np.mean(post ** 2)
    
    if post_energy > 0:
        energy_ratio = np.sqrt(pre_energy / post_energy)
        energy_ratio = np.clip(energy_ratio, 0.5, 2.0)  # Limit correction
        
        # Apply gradually
        t = np.linspace(0, 1, len(post))
        fade_in = t ** 2  # Quadratic fade
        
        result = post * (1.0 - fade_in * 0.6) + post * energy_ratio * fade_in * 0.6
        return result
    
    return post

