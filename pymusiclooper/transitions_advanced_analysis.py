"""
Advanced Segment Analysis - Deep Analysis of Entire Audio Segments.

Extracts maximum information from entire segments for perfect transitions:
- Full-segment rhythm analysis
- Spectral envelope tracking
- Energy flow analysis
- Instrument presence tracking
- Gap detection and masking
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, correlate
from scipy.fft import rfft, irfft, fftfreq
from scipy.ndimage import uniform_filter1d


@dataclass
class SegmentAnalysis:
    """Comprehensive analysis of an audio segment."""
    
    # Rhythm analysis
    beats: np.ndarray  # Beat positions in samples
    beat_intervals: np.ndarray  # Intervals between beats
    rhythm_consistency: float  # How consistent the rhythm is
    tempo: float  # Detected tempo
    
    # Spectral analysis
    spectral_envelope: np.ndarray  # Spectral envelope over time
    spectral_centroid_track: np.ndarray  # Centroid over time
    spectral_flux_track: np.ndarray  # Flux over time
    
    # Energy analysis
    energy_envelope: np.ndarray  # Energy envelope
    energy_peaks: np.ndarray  # Energy peaks
    energy_valleys: np.ndarray  # Energy valleys (good transition points)
    
    # Instrument detection
    has_vocals: bool
    has_ensemble: bool
    has_drums: bool
    instrument_timbre: np.ndarray  # Timbre characteristics
    
    # Gap detection
    rhythm_gaps: np.ndarray  # Detected rhythm gaps
    energy_gaps: np.ndarray  # Detected energy gaps
    
    # Melodic analysis
    pitch_track: np.ndarray  # Pitch over time
    melodic_contour: np.ndarray  # Melodic contour
    
    # Transition points
    optimal_transition_points: np.ndarray  # Best points for transitions


def analyze_segment_comprehensive(
    audio: np.ndarray,
    sr: int,
    start_sample: int = 0,
    end_sample: int | None = None,
) -> SegmentAnalysis:
    """
    Comprehensive analysis of entire audio segment.
    
    Extracts maximum information for perfect transition matching.
    """
    if end_sample is None:
        end_sample = len(audio)
    
    segment = audio[start_sample:end_sample]
    
    if len(segment) < 64:
        # Too short, return minimal analysis
        return _minimal_analysis(segment, sr)
    
    # 1. Rhythm analysis (full segment)
    rhythm_analysis = _analyze_rhythm_comprehensive(segment, sr)
    
    # 2. Spectral analysis (full segment)
    spectral_analysis = _analyze_spectral_comprehensive(segment, sr)
    
    # 3. Energy analysis (full segment)
    energy_analysis = _analyze_energy_comprehensive(segment, sr)
    
    # 4. Instrument detection (full segment)
    instrument_analysis = _detect_instruments_comprehensive(segment, sr)
    
    # 5. Gap detection
    gap_analysis = _detect_gaps_comprehensive(
        segment, sr, rhythm_analysis, energy_analysis
    )
    
    # 6. Melodic analysis
    melodic_analysis = _analyze_melody_comprehensive(segment, sr)
    
    # 7. Find optimal transition points
    optimal_points = _find_optimal_transition_points(
        rhythm_analysis, energy_analysis, gap_analysis, instrument_analysis
    )
    
    return SegmentAnalysis(
        beats=rhythm_analysis['beats'],
        beat_intervals=rhythm_analysis['intervals'],
        rhythm_consistency=rhythm_analysis['consistency'],
        tempo=rhythm_analysis['tempo'],
        spectral_envelope=spectral_analysis['envelope'],
        spectral_centroid_track=spectral_analysis['centroid_track'],
        spectral_flux_track=spectral_analysis['flux_track'],
        energy_envelope=energy_analysis['envelope'],
        energy_peaks=energy_analysis['peaks'],
        energy_valleys=energy_analysis['valleys'],
        has_vocals=instrument_analysis['has_vocals'],
        has_ensemble=instrument_analysis['has_ensemble'],
        has_drums=instrument_analysis['has_drums'],
        instrument_timbre=instrument_analysis['timbre'],
        rhythm_gaps=gap_analysis['rhythm_gaps'],
        energy_gaps=gap_analysis['energy_gaps'],
        pitch_track=melodic_analysis['pitch'],
        melodic_contour=melodic_analysis['contour'],
        optimal_transition_points=optimal_points,
    )


def _analyze_rhythm_comprehensive(audio: np.ndarray, sr: int) -> dict:
    """Comprehensive rhythm analysis of entire segment."""
    # Use autocorrelation for tempo detection
    window_size = min(4096, len(audio) // 4)
    hop = window_size // 2
    
    beats = []
    intervals = []
    tempos = []
    
    for i in range(0, len(audio) - window_size, hop):
        window = audio[i:i+window_size]
        
        # Autocorrelation
        autocorr = correlate(window, window, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        # Find peaks (beat candidates)
        min_period = int(sr / 300)  # Max 300 BPM
        max_period = int(sr / 60)   # Min 60 BPM
        
        if max_period < len(autocorr):
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0:
                peaks, _ = find_peaks(search_range, height=np.max(search_range) * 0.3)
                if len(peaks) > 0:
                    period = peaks[0] + min_period
                    tempo = sr / period * 60
                    tempos.append(tempo)
                    beats.append(i + period)
                    
                    if len(beats) > 1:
                        intervals.append(beats[-1] - beats[-2])
    
    beats = np.array(beats) if beats else np.array([])
    intervals = np.array(intervals) if intervals else np.array([])
    
    # Rhythm consistency
    if len(intervals) > 1:
        mean_interval = np.mean(intervals)
        if mean_interval > 0:
            consistency = 1.0 - (np.std(intervals) / (mean_interval + 1e-10))
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.5
    else:
        consistency = 0.5
    
    tempo = np.median(tempos) if tempos else 120.0
    
    return {
        'beats': beats,
        'intervals': intervals,
        'consistency': consistency,
        'tempo': float(tempo),
    }


def _analyze_spectral_comprehensive(audio: np.ndarray, sr: int) -> dict:
    """Comprehensive spectral analysis of entire segment."""
    window_size = 2048
    hop = window_size // 4
    
    envelope = []
    centroid_track = []
    flux_track = []
    
    prev_spectrum = None
    
    for i in range(0, len(audio) - window_size, hop):
        window = audio[i:i+window_size] * np.hanning(window_size)
        
        # FFT
        n_fft = 2 ** int(np.ceil(np.log2(window_size)))
        fft = rfft(window, n=n_fft)
        magnitude = np.abs(fft)
        
        # Spectral envelope (total energy)
        envelope.append(np.sum(magnitude ** 2))
        
        # Spectral centroid
        freqs = fftfreq(n_fft, 1/sr)[:len(fft)]
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            centroid_track.append(centroid / (sr / 2))  # Normalize
        else:
            centroid_track.append(0.0)
        
        # Spectral flux
        if prev_spectrum is not None:
            flux = np.sum((magnitude - prev_spectrum) ** 2)
            flux_track.append(flux)
        else:
            flux_track.append(0.0)
        
        prev_spectrum = magnitude
    
    return {
        'envelope': np.array(envelope),
        'centroid_track': np.array(centroid_track),
        'flux_track': np.array(flux_track),
    }


def _analyze_energy_comprehensive(audio: np.ndarray, sr: int) -> dict:
    """Comprehensive energy analysis of entire segment."""
    window_size = int(sr * 0.05)  # 50ms windows
    hop = window_size // 2
    
    energy = []
    
    for i in range(0, len(audio) - window_size, hop):
        window = audio[i:i+window_size]
        energy.append(np.mean(window ** 2))
    
    energy = np.array(energy)
    
    # Smooth envelope
    envelope = uniform_filter1d(energy, size=5)
    
    # Find peaks and valleys
    if len(envelope) > 10:
        peaks, _ = find_peaks(envelope, height=np.max(envelope) * 0.3)
        valleys, _ = find_peaks(-envelope, height=-np.min(envelope) * 0.7)
    else:
        peaks = np.array([])
        valleys = np.array([])
    
    # Convert to sample indices
    peak_samples = peaks * hop if len(peaks) > 0 else np.array([])
    valley_samples = valleys * hop if len(valleys) > 0 else np.array([])
    
    return {
        'envelope': envelope,
        'peaks': peak_samples,
        'valleys': valley_samples,
    }


def _detect_instruments_comprehensive(audio: np.ndarray, sr: int) -> dict:
    """Comprehensive instrument detection."""
    window_size = 2048
    hop = window_size // 4
    
    if len(audio) < window_size:
        # Too short, return minimal
        return {
            'has_vocals': False,
            'has_ensemble': False,
            'has_drums': False,
            'timbre': np.zeros(13),
        }
    
    vocal_scores = []
    ensemble_scores = []
    drum_scores = []
    timbre_features = []
    
    for i in range(0, len(audio) - window_size, hop):
        window = audio[i:i+window_size] * np.hanning(window_size)
        
        # FFT
        n_fft = 2 ** int(np.ceil(np.log2(window_size)))
        fft = rfft(window, n=n_fft)
        magnitude = np.abs(fft)
        freqs = fftfreq(n_fft, 1/sr)[:len(fft)]
        
        if len(freqs) == 0 or len(magnitude) == 0:
            continue
        
        # Vocal detection (200-2000 Hz)
        vocal_mask = (freqs >= 200) & (freqs <= 2000)
        if np.any(vocal_mask):
            vocal_energy = np.sum(magnitude[vocal_mask])
            total_energy = np.sum(magnitude)
            if total_energy > 0:
                vocal_scores.append(vocal_energy / total_energy)
            else:
                vocal_scores.append(0.0)
        else:
            vocal_scores.append(0.0)
        
        # Ensemble detection (200-3000 Hz)
        ensemble_mask = (freqs >= 200) & (freqs <= 3000)
        if np.any(ensemble_mask):
            ensemble_energy = np.sum(magnitude[ensemble_mask])
            total_energy = np.sum(magnitude)
            if total_energy > 0:
                ensemble_scores.append(ensemble_energy / total_energy)
            else:
                ensemble_scores.append(0.0)
        else:
            ensemble_scores.append(0.0)
        
        # Drum detection (transients, wide frequency range)
        drum_mask = (freqs >= 50) & (freqs <= 8000)
        if np.any(drum_mask):
            drum_energy = np.sum(magnitude[drum_mask])
            total_energy = np.sum(magnitude)
            if total_energy > 0:
                # Check for transient characteristics
                drum_mag = magnitude[drum_mask]
                if len(drum_mag) > 1:
                    transient_score = np.max(np.abs(np.diff(drum_mag)))
                else:
                    transient_score = 0.0
                drum_scores.append((drum_energy / total_energy) * (1.0 + transient_score * 0.1))
            else:
                drum_scores.append(0.0)
        else:
            drum_scores.append(0.0)
        
        # Timbre features (MFCC-like)
        # Use mel-scale energy bands
        n_mels = 13
        mel_energies = []
        for mel_idx in range(n_mels):
            mel_low = 200 + mel_idx * 150
            mel_high = mel_low + 150
            mel_mask = (freqs >= mel_low) & (freqs < mel_high)
            if np.any(mel_mask):
                mel_energies.append(np.sum(magnitude[mel_mask]))
            else:
                mel_energies.append(0.0)
        
        timbre_features.append(mel_energies)
    
    # Aggregate (with safety checks)
    has_vocals = (np.mean(vocal_scores) > 0.3) if len(vocal_scores) > 0 else False
    has_ensemble = (np.mean(ensemble_scores) > 0.25) if len(ensemble_scores) > 0 else False
    has_drums = (np.mean(drum_scores) > 0.4) if len(drum_scores) > 0 else False
    
    if timbre_features and len(timbre_features) > 0:
        timbre = np.mean(timbre_features, axis=0)
    else:
        timbre = np.zeros(13)
    
    return {
        'has_vocals': has_vocals,
        'has_ensemble': has_ensemble,
        'has_drums': has_drums,
        'timbre': timbre,
    }


def _detect_gaps_comprehensive(
    audio: np.ndarray,
    sr: int,
    rhythm_analysis: dict,
    energy_analysis: dict,
) -> dict:
    """Detect rhythm and energy gaps in segment."""
    # Rhythm gaps
    rhythm_gaps = []
    intervals = rhythm_analysis['intervals']
    if len(intervals) > 1:
        mean_interval = np.mean(intervals)
        if mean_interval > 0:
            std_interval = np.std(intervals)
            
            for i, interval in enumerate(intervals):
                if interval > mean_interval + std_interval * 1.5:
                    # Gap detected
                    gap_start = rhythm_analysis['beats'][i] if i < len(rhythm_analysis['beats']) else 0
                    gap_end = rhythm_analysis['beats'][i+1] if i+1 < len(rhythm_analysis['beats']) else len(audio)
                    rhythm_gaps.append((gap_start, gap_end, interval - mean_interval))
    
    # Energy gaps
    energy_gaps = []
    envelope = energy_analysis['envelope']
    if len(envelope) > 1:
        mean_energy = np.mean(envelope)
        if mean_energy > 0:
            std_energy = np.std(envelope)
            
            for i, energy in enumerate(envelope):
                if energy < mean_energy - std_energy * 1.0:
                    # Energy gap detected
                    gap_sample = i * (len(audio) // len(envelope))
                    energy_gaps.append(gap_sample)
    
    return {
        'rhythm_gaps': np.array(rhythm_gaps) if rhythm_gaps else np.array([]),
        'energy_gaps': np.array(energy_gaps) if energy_gaps else np.array([]),
    }


def _analyze_melody_comprehensive(audio: np.ndarray, sr: int) -> dict:
    """Comprehensive melodic analysis."""
    window_size = 2048
    hop = window_size // 4
    
    pitch_track = []
    contour = []
    
    for i in range(0, len(audio) - window_size, hop):
        window = audio[i:i+window_size] * np.hanning(window_size)
        
        # Autocorrelation for pitch
        autocorr = correlate(window, window, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        # Find fundamental
        min_period = int(sr / 1000)  # Max 1000 Hz
        max_period = int(sr / 80)    # Min 80 Hz
        
        if max_period < len(autocorr):
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0:
                peak_idx = np.argmax(search_range) + min_period
                if peak_idx > 0:
                    pitch = sr / peak_idx
                    pitch_track.append(pitch)
                else:
                    pitch_track.append(0.0)
            else:
                pitch_track.append(0.0)
        else:
            pitch_track.append(0.0)
    
    pitch_track = np.array(pitch_track)
    
    # Melodic contour (pitch direction)
    if len(pitch_track) > 1:
        contour = np.diff(pitch_track)
    else:
        contour = np.array([])
    
    return {
        'pitch': pitch_track,
        'contour': contour,
    }


def _find_optimal_transition_points(
    rhythm_analysis: dict,
    energy_analysis: dict,
    gap_analysis: dict,
    instrument_analysis: dict,
) -> np.ndarray:
    """Find optimal transition points based on comprehensive analysis."""
    optimal_points = []
    
    # Prefer points at energy valleys
    if len(energy_analysis['valleys']) > 0:
        optimal_points.extend(energy_analysis['valleys'].tolist())
    
    # Prefer points at beats (if rhythm is consistent)
    if rhythm_analysis['consistency'] > 0.7 and len(rhythm_analysis['beats']) > 0:
        optimal_points.extend(rhythm_analysis['beats'].tolist())
    
    # Avoid gaps
    if len(gap_analysis['rhythm_gaps']) > 0:
        # Remove points in gaps
        optimal_points = [p for p in optimal_points if not _point_in_gap(p, gap_analysis['rhythm_gaps'])]
    
    return np.array(optimal_points) if optimal_points else np.array([])


def _point_in_gap(point: int, gaps: np.ndarray) -> bool:
    """Check if point is in a gap."""
    for gap in gaps:
        if len(gap) >= 2 and gap[0] <= point <= gap[1]:
            return True
    return False


def _minimal_analysis(audio: np.ndarray, sr: int) -> SegmentAnalysis:
    """Minimal analysis for very short segments."""
    return SegmentAnalysis(
        beats=np.array([]),
        beat_intervals=np.array([]),
        rhythm_consistency=0.5,
        tempo=120.0,
        spectral_envelope=np.array([np.mean(audio ** 2)]),
        spectral_centroid_track=np.array([0.3]),
        spectral_flux_track=np.array([0.0]),
        energy_envelope=np.array([np.mean(audio ** 2)]),
        energy_peaks=np.array([]),
        energy_valleys=np.array([]),
        has_vocals=False,
        has_ensemble=False,
        has_drums=False,
        instrument_timbre=np.zeros(13),
        rhythm_gaps=np.array([]),
        energy_gaps=np.array([]),
        pitch_track=np.array([]),
        melodic_contour=np.array([]),
        optimal_transition_points=np.array([]),
    )

