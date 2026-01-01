"""
Audio Feature Extraction Module.

Extracts comprehensive audio features for loop detection:
- Spectral: STFT, Mel spectrogram, MFCC
- Harmonic: Chroma CENS, Chroma CQT, Tonnetz
- Rhythm: Beat tracking, onset detection, tempo
- Dynamics: RMS, loudness, spectral flux
- Perceptual: Masking curves, A-weighting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d

if TYPE_CHECKING:
    from pymusiclooper.audio import MLAudio

import librosa

from pymusiclooper.exceptions import LoopNotFoundError

# Constants
HOP_LENGTH = 512
N_FFT = 2048


@dataclass(slots=True, frozen=True)
class Features:
    """Immutable container for all extracted audio features."""
    
    # Core features (2D arrays: [n_features, n_frames])
    chroma_cens: np.ndarray      # Chroma Energy Normalized Statistics
    chroma_cqt: np.ndarray       # Chroma from Constant-Q Transform
    tonnetz: np.ndarray          # Tonal centroid features
    mfcc: np.ndarray             # Mel-frequency cepstral coefficients
    
    # 1D arrays - spectral characteristics
    rms: np.ndarray              # Root mean square energy
    spectral_centroid: np.ndarray  # Spectral center of mass
    spectral_bandwidth: np.ndarray  # Spectral spread
    spectral_flatness: np.ndarray   # Noise vs tonal
    spectral_flux: np.ndarray       # Frame-to-frame spectral change
    spectral_rolloff: np.ndarray    # Frequency below which 85% energy
    spectral_contrast: np.ndarray   # Valley-to-peak differences per band
    
    # 1D arrays - rhythm/timing
    onset_env: np.ndarray        # Onset strength envelope
    onset_peaks: np.ndarray      # Binary onset locations
    transient_strength: np.ndarray  # Combined transient detection
    beat_phase: np.ndarray       # Phase within beat (0-1)
    sub_beat_phase: np.ndarray   # Phase within 16th note (0-1)
    
    # Rhythm info
    beats: np.ndarray            # Beat frame indices
    downbeats: np.ndarray        # Downbeat frame indices
    bpm: float                   # Tempo in BPM
    bar_length: int              # Frames per bar
    beat_length: int             # Frames per beat
    
    # Self-similarity matrices (precomputed)
    ssm_chroma: np.ndarray       # Chroma-based SSM
    ssm_mfcc: np.ndarray         # MFCC-based SSM
    
    # Power/loudness (2D)
    power_db: np.ndarray         # A-weighted power spectrogram in dB
    mel_spectrogram: np.ndarray  # Mel spectrogram
    
    # Perceptual
    masking_curve: np.ndarray    # Temporal masking estimate
    loudness: np.ndarray         # Perceptual loudness (approximation)
    
    # Raw audio for sample-level operations
    audio: np.ndarray
    sr: int
    hop: int
    n_frames: int
    duration: float


def extract_features(
    mlaudio: MLAudio,
    approx_start: float | None,
    approx_end: float | None,
    brute_force: bool,
) -> Features:
    """
    Extract comprehensive audio features for loop detection.
    
    Uses single STFT computation with derived features for efficiency.
    """
    audio = mlaudio.audio.astype(np.float32)
    sr = mlaudio.rate
    hop = HOP_LENGTH
    duration = mlaudio.total_duration

    # === Single STFT computation (reuse for all spectral features) ===
    S = librosa.stft(y=audio, n_fft=N_FFT, hop_length=hop)
    S_mag = np.abs(S) + 1e-10
    S_power = S_mag ** 2
    n_frames = S_mag.shape[1]

    # === Chroma features ===
    chroma_cens = librosa.feature.chroma_cens(
        y=audio, sr=sr, hop_length=hop
    ).astype(np.float32)
    
    try:
        C = librosa.cqt(y=audio, sr=sr, hop_length=hop)
        chroma_cqt = librosa.feature.chroma_cqt(
            C=np.abs(C), sr=sr
        ).astype(np.float32)
    except Exception:
        chroma_cqt = chroma_cens.copy()
    
    # Tonnetz from normalized chroma
    chroma_norm = librosa.util.normalize(chroma_cens, norm=1, axis=0)
    chroma_norm = np.nan_to_num(chroma_norm, nan=0.0)
    tonnetz = librosa.feature.tonnetz(
        chroma=chroma_norm, sr=sr
    ).astype(np.float32)
    tonnetz = np.nan_to_num(tonnetz, nan=0.0)

    # === Mel/MFCC features ===
    mel = librosa.feature.melspectrogram(
        S=S_power, sr=sr, n_mels=128, hop_length=hop
    )
    mel_safe = mel + 1e-10
    mel_ref = max(np.max(mel_safe), 1e-10)
    mel_db = librosa.power_to_db(mel_safe, ref=mel_ref)
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=20).astype(np.float32)

    # === Spectral features ===
    spectral_centroid = librosa.feature.spectral_centroid(
        S=S_mag, sr=sr, hop_length=hop
    )[0]
    spectral_centroid = (spectral_centroid / (sr / 2 + 1e-10)).astype(np.float32)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        S=S_mag, sr=sr, hop_length=hop
    )[0]
    spectral_bandwidth = (spectral_bandwidth / (sr / 2 + 1e-10)).astype(np.float32)
    
    spectral_flatness = librosa.feature.spectral_flatness(S=S_mag)[0].astype(np.float32)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=S_mag, sr=sr, hop_length=hop, roll_percent=0.85
    )[0]
    spectral_rolloff = (spectral_rolloff / (sr / 2 + 1e-10)).astype(np.float32)
    
    # Spectral contrast (per-band valley-to-peak)
    spectral_contrast = librosa.feature.spectral_contrast(
        S=S_mag, sr=sr, hop_length=hop
    ).astype(np.float32)

    # === RMS and dynamics ===
    rms = librosa.feature.rms(S=S_mag, hop_length=hop)[0]
    rms_max = np.max(rms) + 1e-10
    rms = (rms / rms_max).astype(np.float32)

    # === Spectral flux ===
    S_diff = np.diff(S_mag, axis=1, prepend=S_mag[:, :1])
    spectral_flux = np.sqrt(np.mean(np.maximum(S_diff, 0) ** 2, axis=0))
    flux_max = np.max(spectral_flux) + 1e-10
    spectral_flux = (spectral_flux / flux_max).astype(np.float32)

    # === Onset detection ===
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=hop)
    onset_max = np.max(onset_env) + 1e-10
    onset_env = (onset_env / onset_max).astype(np.float32)
    
    onset_peaks = np.zeros(n_frames, dtype=np.float32)
    peak_idx = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False
    )
    onset_peaks[peak_idx[peak_idx < n_frames]] = 1.0

    # === Transient strength ===
    hf_energy = np.mean(S_power[S_power.shape[0] // 2:, :], axis=0)
    hf_max = np.max(hf_energy) + 1e-10
    hf_energy = hf_energy / hf_max
    transient_strength = (
        onset_env * 0.5 + hf_energy * 0.3 + spectral_flux * 0.2
    ).astype(np.float32)
    ts_max = np.max(transient_strength) + 1e-10
    transient_strength = transient_strength / ts_max

    # === Rhythm analysis ===
    bpm, beats, downbeats, beat_phase = _analyze_rhythm(
        onset_env, sr, hop, mlaudio, n_frames, approx_start, approx_end, brute_force
    )
    
    beat_length = max(1, int(np.median(np.diff(beats))) if len(beats) > 1 
                      else mlaudio.seconds_to_frames(60.0 / max(bpm, 1)))
    bar_length = beat_length * 4
    
    sub_beat_phase = _compute_sub_beat_phase(beats, n_frames)

    # === Perceptual features ===
    # Masking curve (psychoacoustic temporal masking estimate)
    masking = (
        rms * 0.5 + 
        spectral_flatness * 0.3 + 
        (1 - spectral_flux) * 0.2
    ).astype(np.float32)
    masking = uniform_filter1d(masking, size=11)
    
    # Perceptual loudness (A-weighted power)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freqs_safe = np.maximum(freqs, 1e-10)
    S_weighted = librosa.perceptual_weighting(S=S_power, frequencies=freqs_safe)
    S_weighted_safe = S_weighted + 1e-10
    S_ref = max(np.median(S_weighted_safe), np.max(S_weighted_safe) * 0.01, 1e-10)
    power_db = librosa.power_to_db(S_weighted_safe, ref=S_ref).astype(np.float32)
    
    # Loudness approximation (sum of weighted power)
    loudness = np.mean(power_db, axis=0).astype(np.float32)
    loudness = (loudness - np.min(loudness)) / (np.max(loudness) - np.min(loudness) + 1e-10)

    # === Self-similarity matrices ===
    ssm_chroma, ssm_mfcc = _compute_ssm(chroma_cens, mfcc, beats)

    return Features(
        chroma_cens=chroma_cens,
        chroma_cqt=chroma_cqt,
        tonnetz=tonnetz,
        mfcc=mfcc,
        rms=rms,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_flatness=spectral_flatness,
        spectral_flux=spectral_flux,
        spectral_rolloff=spectral_rolloff,
        spectral_contrast=spectral_contrast,
        onset_env=onset_env,
        onset_peaks=onset_peaks,
        transient_strength=transient_strength,
        beat_phase=beat_phase,
        sub_beat_phase=sub_beat_phase,
        beats=beats,
        downbeats=downbeats,
        bpm=bpm,
        bar_length=bar_length,
        beat_length=beat_length,
        ssm_chroma=ssm_chroma,
        ssm_mfcc=ssm_mfcc,
        power_db=power_db,
        mel_spectrogram=mel_db.astype(np.float32),
        masking_curve=masking,
        loudness=loudness,
        audio=audio,
        sr=sr,
        hop=hop,
        n_frames=n_frames,
        duration=duration,
    )


def _compute_ssm(
    chroma: np.ndarray, 
    mfcc: np.ndarray, 
    beats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute beat-synchronized self-similarity matrices."""
    if len(beats) > 2:
        chroma_beat = librosa.util.sync(chroma, beats, aggregate=np.median)
        mfcc_beat = librosa.util.sync(mfcc, beats, aggregate=np.median)
    else:
        chroma_beat = chroma
        mfcc_beat = mfcc
    
    ssm_chroma = _ssm_normalized(chroma_beat)
    ssm_mfcc = _ssm_normalized(mfcc_beat)
    
    return ssm_chroma.astype(np.float32), ssm_mfcc.astype(np.float32)


@njit(cache=True, fastmath=True)
def _ssm_normalized(features: np.ndarray) -> np.ndarray:
    """Compute normalized self-similarity matrix using cosine similarity."""
    n_feat, n_frames = features.shape
    norms = np.zeros(n_frames, dtype=np.float64)
    
    for i in range(n_frames):
        s = 0.0
        for k in range(n_feat):
            s += features[k, i] ** 2
        norms[i] = np.sqrt(s) if s > 1e-16 else 1.0
    
    ssm = np.zeros((n_frames, n_frames), dtype=np.float64)
    for i in range(n_frames):
        for j in range(i, n_frames):
            dot = 0.0
            for k in range(n_feat):
                dot += features[k, i] * features[k, j]
            sim = dot / (norms[i] * norms[j])
            ssm[i, j] = sim
            ssm[j, i] = sim
    
    return ssm


def _analyze_rhythm(
    onset_env: np.ndarray,
    sr: int,
    hop: int,
    mlaudio: MLAudio,
    n_frames: int,
    approx_start: float | None,
    approx_end: float | None,
    brute_force: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Analyze rhythm: tempo, beats, downbeats, beat phase."""
    
    if approx_start is not None and approx_end is not None:
        # Approximate mode: only search near specified points
        n_check = mlaudio.seconds_to_frames(2)
        start_f = mlaudio.seconds_to_frames(approx_start, apply_trim_offset=True)
        end_f = mlaudio.seconds_to_frames(approx_end, apply_trim_offset=True)
        beats = np.concatenate([
            np.arange(max(0, start_f - n_check), min(n_frames, start_f + n_check)),
            np.arange(max(0, end_f - n_check), min(n_frames, end_f + n_check)),
        ]).astype(np.int64)
        beat_phase = np.zeros(n_frames, dtype=np.float32)
        return 120.0, beats, beats[::4], beat_phase

    if brute_force:
        # Brute force: every frame is a candidate
        beats = np.arange(n_frames, dtype=np.int64)
        beat_phase = np.zeros(n_frames, dtype=np.float32)
        return 120.0, beats, beats[::4], beat_phase

    try:
        # Normal beat tracking
        bpm, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop
        )
        bpm = float(bpm[0]) if isinstance(bpm, np.ndarray) else float(bpm)
        bpm = max(bpm, 1.0)
        
        # PLP for additional beat refinement
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=hop)
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        
        beats = np.union1d(beats, beats_plp).astype(np.int64)
        beats = beats[beats < n_frames]
        beats.sort()
        
        downbeats = beats[::4] if len(beats) >= 4 else beats
        beat_phase = _compute_beat_phase(beats, n_frames)

        logging.info(f"Detected {len(beats)} beats at {bpm:.0f} BPM")
        return bpm, beats, downbeats, beat_phase

    except Exception as e:
        raise LoopNotFoundError(f"Rhythm analysis failed: {e}") from e


@njit(cache=True, fastmath=True)
def _compute_beat_phase(beats: np.ndarray, n_frames: int) -> np.ndarray:
    """Compute phase within beat cycle (0=on beat, 0.5=off beat)."""
    phase = np.zeros(n_frames, dtype=np.float32)
    n_beats = len(beats)
    
    if n_beats < 2:
        return phase
    
    for i in range(n_beats - 1):
        start, end = beats[i], beats[i + 1]
        length = end - start
        if length > 0:
            for j in range(length):
                phase[start + j] = float(j) / float(length)
    
    return phase


@njit(cache=True, fastmath=True)
def _compute_sub_beat_phase(beats: np.ndarray, n_frames: int) -> np.ndarray:
    """Compute phase within 16th note (for micro-timing)."""
    phase = np.zeros(n_frames, dtype=np.float32)
    n_beats = len(beats)
    
    if n_beats < 2:
        return phase
    
    for i in range(n_beats - 1):
        start, end = beats[i], beats[i + 1]
        length = end - start
        if length > 0:
            for j in range(length):
                phase[start + j] = (float(j) / float(length) * 4.0) % 1.0
    
    return phase


def nearest_zero_crossing(audio: np.ndarray, sr: int, target: int) -> int:
    """Find nearest zero crossing to target sample for click-free transitions."""
    search_ms = 5
    search_samples = int(sr * search_ms / 1000)
    
    start = max(0, target - search_samples)
    end = min(len(audio), target + search_samples)
    
    if end <= start:
        return max(0, min(target, len(audio) - 1))
    
    if len(audio.shape) == 2:
        mono = np.mean(audio[start:end], axis=1)
    else:
        mono = audio[start:end]
    
    if len(mono) < 2:
        return target
    
    signs = np.sign(mono)
    crossings = np.where(np.diff(signs) != 0)[0]
    
    if len(crossings) == 0:
        return target
    
    relative_target = target - start
    closest_idx = crossings[np.argmin(np.abs(crossings - relative_target))]
    
    return start + closest_idx

