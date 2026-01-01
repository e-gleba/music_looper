"""
Ultra-melodic loop detection with optimized performance.

Features:
- Vectorized numpy operations
- Numba JIT compilation for hot paths
- Cross-correlation for sample-accurate transitions
- Psychoacoustic transition model
- Spectral phase coherence
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, find_peaks

if TYPE_CHECKING:
    from pymusiclooper.audio import MLAudio

import librosa

from pymusiclooper.exceptions import LoopNotFoundError

# Constants
HOP_LENGTH = 512
N_FFT = 2048

# Krumhansl-Kessler key profiles (for tonal analysis)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)


@dataclass(slots=True)
class LoopPair:
    """Single loop point with optimized sample alignment."""
    _loop_start_frame_idx: int
    _loop_end_frame_idx: int
    note_distance: float
    loudness_difference: float
    score: float = 0.0
    loop_start: int = 0
    loop_end: int = 0


@dataclass(slots=True, frozen=True)
class Features:
    """Immutable audio features container."""
    # Core features (2D arrays)
    chroma_cens: np.ndarray
    chroma_cqt: np.ndarray
    tonnetz: np.ndarray
    mfcc: np.ndarray
    
    # 1D arrays - dynamics/spectral
    rms: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_flatness: np.ndarray
    spectral_flux: np.ndarray
    
    # 1D arrays - rhythm
    onset_env: np.ndarray
    onset_peaks: np.ndarray
    transient_strength: np.ndarray
    beat_phase: np.ndarray
    sub_beat_phase: np.ndarray
    
    # Rhythm info
    beats: np.ndarray
    downbeats: np.ndarray
    bpm: float
    bar_length: int
    beat_length: int
    
    # SSM matrices (precomputed)
    ssm_chroma: np.ndarray
    ssm_mfcc: np.ndarray
    
    # Power/loudness
    power_db: np.ndarray
    masking_curve: np.ndarray
    
    # Raw audio for sample-level operations
    audio: np.ndarray
    sr: int
    hop: int
    n_frames: int
    duration: float


def find_best_loop_points(
    mlaudio: MLAudio,
    min_duration_multiplier: float = 0.35,
    min_loop_duration: float | None = None,
    max_loop_duration: float | None = None,
    approx_loop_start: float | None = None,
    approx_loop_end: float | None = None,
    brute_force: bool = False,
    disable_pruning: bool = False,
) -> list[LoopPair]:
    """Find melodically perfect loop points with optimized performance."""
    t0 = time.perf_counter()

    min_frames = (
        mlaudio.seconds_to_frames(min_loop_duration)
        if min_loop_duration else
        mlaudio.seconds_to_frames(int(min_duration_multiplier * mlaudio.total_duration))
    )
    min_frames = max(1, min_frames)

    max_frames = (
        mlaudio.seconds_to_frames(max_loop_duration)
        if max_loop_duration else
        mlaudio.seconds_to_frames(mlaudio.total_duration)
    )

    # Extract features (optimized)
    feat = extract_features(mlaudio, approx_loop_start, approx_loop_end, brute_force)
    logging.info(f"Feature extraction: {time.perf_counter() - t0:.3f}s")

    # Generate candidates (vectorized)
    t1 = time.perf_counter()
    candidates = _find_candidates_fast(feat, min_frames, max_frames)
    logging.info(f"Found {len(candidates)} candidates in {time.perf_counter() - t1:.3f}s")

    if not candidates:
        raise LoopNotFoundError(f'No loops found for "{mlaudio.filename}".')

    # Score with optimized vectorized operations
    scored = _score_vectorized(mlaudio, feat, candidates, disable_pruning)

    # Sort by score DESCENDING
    scored.sort(key=lambda x: x.score, reverse=True)

    if len(scored) > 1:
        _optimize_selection(scored, feat)

    # Finalize with sample-accurate alignment
    _finalize_with_alignment(mlaudio, feat, scored)

    if not scored:
        raise LoopNotFoundError(f'No loops found for "{mlaudio.filename}".')

    scored.sort(key=lambda x: x.score, reverse=True)

    logging.info(f"Final {len(scored)} loops in {time.perf_counter() - t0:.3f}s")
    logging.info(f"Best score: {scored[0].score:.4f}")
    return scored


def extract_features(
    mlaudio: MLAudio,
    approx_start: float | None,
    approx_end: float | None,
    brute_force: bool,
) -> Features:
    """Optimized feature extraction with minimal redundant computation."""
    audio = mlaudio.audio.astype(np.float32)
    sr = mlaudio.rate
    hop = HOP_LENGTH
    duration = mlaudio.total_duration

    # Single STFT computation (reuse for multiple features)
    S = librosa.stft(y=audio, n_fft=N_FFT, hop_length=hop)
    S_mag = np.abs(S) + 1e-10
    S_power = S_mag ** 2
    n_frames = S_mag.shape[1]

    # === Chroma features (batch) ===
    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr, hop_length=hop).astype(np.float32)
    
    try:
        C = librosa.cqt(y=audio, sr=sr, hop_length=hop)
        chroma_cqt = librosa.feature.chroma_cqt(C=np.abs(C), sr=sr).astype(np.float32)
    except Exception:
        chroma_cqt = chroma_cens.copy()
    
    # Tonnetz from normalized chroma
    chroma_norm = librosa.util.normalize(chroma_cens, norm=1, axis=0)
    chroma_norm = np.nan_to_num(chroma_norm, nan=0.0)
    tonnetz = librosa.feature.tonnetz(chroma=chroma_norm, sr=sr).astype(np.float32)
    tonnetz = np.nan_to_num(tonnetz, nan=0.0)

    # === MFCC (from mel spectrogram) ===
    mel = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=128, hop_length=hop)
    mel_safe = mel + 1e-10
    mel_ref = max(np.max(mel_safe), 1e-10)
    mel_db = librosa.power_to_db(mel_safe, ref=mel_ref)
    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=20).astype(np.float32)

    # === Spectral features (vectorized) ===
    spectral_centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, hop_length=hop)[0]
    spectral_centroid = (spectral_centroid / (sr / 2 + 1e-10)).astype(np.float32)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr, hop_length=hop)[0]
    spectral_bandwidth = (spectral_bandwidth / (sr / 2 + 1e-10)).astype(np.float32)
    
    spectral_flatness = librosa.feature.spectral_flatness(S=S_mag)[0].astype(np.float32)

    # === RMS and dynamics ===
    rms = librosa.feature.rms(S=S_mag, hop_length=hop)[0]
    rms_max = np.max(rms) + 1e-10
    rms = (rms / rms_max).astype(np.float32)

    # === Spectral flux (vectorized) ===
    S_diff = np.diff(S_mag, axis=1, prepend=S_mag[:, :1])
    spectral_flux = np.sqrt(np.mean(np.maximum(S_diff, 0) ** 2, axis=0))
    flux_max = np.max(spectral_flux) + 1e-10
    spectral_flux = (spectral_flux / flux_max).astype(np.float32)

    # === Onset detection ===
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=hop)
    onset_max = np.max(onset_env) + 1e-10
    onset_env = (onset_env / onset_max).astype(np.float32)
    
    # Onset peaks (binary)
    onset_peaks = np.zeros(n_frames, dtype=np.float32)
    peak_idx = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False)
    onset_peaks[peak_idx[peak_idx < n_frames]] = 1.0

    # === Transient strength (combined features) ===
    hf_energy = np.mean(S_power[S_power.shape[0] // 2:, :], axis=0)
    hf_max = np.max(hf_energy) + 1e-10
    hf_energy = hf_energy / hf_max
    transient_strength = (onset_env * 0.5 + hf_energy * 0.3 + spectral_flux * 0.2).astype(np.float32)
    ts_max = np.max(transient_strength) + 1e-10
    transient_strength = transient_strength / ts_max

    # === Rhythm analysis ===
    bpm, beats, downbeats, beat_phase = _analyze_rhythm_fast(
        onset_env, sr, hop, mlaudio, n_frames, approx_start, approx_end, brute_force
    )
    
    beat_length = max(1, int(np.median(np.diff(beats))) if len(beats) > 1 else mlaudio.seconds_to_frames(60.0 / max(bpm, 1)))
    bar_length = beat_length * 4
    
    # Sub-beat phase
    sub_beat_phase = _compute_sub_beat_phase_fast(beats, n_frames)

    # === Perceptual masking curve ===
    masking = (rms * 0.5 + spectral_flatness * 0.3 + (1 - spectral_flux) * 0.2).astype(np.float32)
    masking = uniform_filter1d(masking, size=11)

    # === Power (for loudness matching) ===
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freqs_safe = np.maximum(freqs, 1e-10)
    S_weighted = librosa.perceptual_weighting(S=S_power, frequencies=freqs_safe)
    S_weighted_safe = S_weighted + 1e-10
    S_ref = max(np.median(S_weighted_safe), np.max(S_weighted_safe) * 0.01, 1e-10)
    power_db = librosa.power_to_db(S_weighted_safe, ref=S_ref).astype(np.float32)

    # === Self-similarity matrices (precomputed, downsampled) ===
    ssm_chroma, ssm_mfcc = _compute_ssm_fast(chroma_cens, mfcc, beats)

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
        masking_curve=masking,
        audio=audio,
        sr=sr,
        hop=hop,
        n_frames=n_frames,
        duration=duration,
    )


def _compute_ssm_fast(chroma: np.ndarray, mfcc: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast SSM computation with beat synchronization."""
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
    """Normalized self-similarity matrix (numba optimized)."""
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


def _analyze_rhythm_fast(onset_env, sr, hop, mlaudio, n_frames, approx_start, approx_end, brute_force):
    """Fast rhythm analysis."""
    if approx_start is not None and approx_end is not None:
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
        beats = np.arange(n_frames, dtype=np.int64)
        beat_phase = np.zeros(n_frames, dtype=np.float32)
        return 120.0, beats, beats[::4], beat_phase

    try:
        bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
        bpm = float(bpm[0]) if isinstance(bpm, np.ndarray) else float(bpm)
        bpm = max(bpm, 1.0)
        
        # PLP for additional beat detection
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=hop)
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        
        beats = np.union1d(beats, beats_plp).astype(np.int64)
        beats = beats[beats < n_frames]
        beats.sort()
        
        downbeats = beats[::4] if len(beats) >= 4 else beats
        beat_phase = _compute_beat_phase_fast(beats, n_frames)

        logging.info(f"Detected {len(beats)} beats at {bpm:.0f} BPM")
        return bpm, beats, downbeats, beat_phase

    except Exception as e:
        raise LoopNotFoundError(f"Rhythm analysis failed: {e}") from e


@njit(cache=True, fastmath=True)
def _compute_beat_phase_fast(beats: np.ndarray, n_frames: int) -> np.ndarray:
    """Fast beat phase computation."""
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
def _compute_sub_beat_phase_fast(beats: np.ndarray, n_frames: int) -> np.ndarray:
    """Fast sub-beat phase computation (16th notes)."""
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


def _find_candidates_fast(feat: Features, min_frames: int, max_frames: int) -> list[LoopPair]:
    """Optimized candidate generation using vectorized operations."""
    candidates = []
    existing = set()
    
    # Use SSM for primary candidate detection (multiple thresholds for robustness)
    for thresh in [45, 55, 65]:
        candidates.extend(_ssm_candidates_vectorized(feat.ssm_chroma, feat.beats, feat.power_db, min_frames, max_frames, feat.n_frames, thresh))
        candidates.extend(_ssm_candidates_vectorized(feat.ssm_mfcc, feat.beats, feat.power_db, min_frames, max_frames, feat.n_frames, thresh + 5))
    
    # Downbeat-aligned candidates (robust fallback)
    candidates.extend(_downbeat_candidates_fast(feat, min_frames, max_frames))
    
    # Beat-aligned candidates (additional)
    candidates.extend(_beat_candidates_fast(feat, min_frames, max_frames))
    
    # Deduplicate
    unique = []
    for c in candidates:
        key = (c._loop_start_frame_idx, c._loop_end_frame_idx)
        if key not in existing:
            unique.append(c)
            existing.add(key)
    
    return unique


def _beat_candidates_fast(feat: Features, min_frames: int, max_frames: int) -> list[LoopPair]:
    """Beat-aligned candidate generation (robust fallback)."""
    candidates = []
    beats = feat.beats
    n_beats = len(beats)
    power = np.max(feat.power_db, axis=0)
    
    if n_beats < 8:
        return []
    
    # Sample beats with step for efficiency
    step = max(1, n_beats // 150)
    
    # Calculate max offset in beats that could satisfy min_frames
    # We need to find beats that span at least min_frames
    for i in range(0, n_beats - 8, step):
        for j in range(i + 8, n_beats, step):
            loop_len = beats[j] - beats[i]
            
            if loop_len < min_frames:
                continue
            if loop_len > max_frames:
                break  # No point checking further
            
            # Quick chroma check
            f1, f2 = min(int(beats[i]), feat.n_frames - 1), min(int(beats[j]), feat.n_frames - 1)
            c1 = feat.chroma_cens[:, f1]
            c2 = feat.chroma_cens[:, f2]
            
            sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
            
            if sim < 0.4:  # Relaxed threshold
                continue
            
            ps = power[min(int(beats[i]), len(power) - 1)]
            pe = power[max(0, min(int(beats[j]) - 1, len(power) - 1))]
            
            candidates.append(LoopPair(
                _loop_start_frame_idx=int(beats[i]),
                _loop_end_frame_idx=int(beats[j]),
                note_distance=1.0 - sim,
                loudness_difference=abs(ps - pe),
            ))
    
    return candidates


def _ssm_candidates_vectorized(ssm, beats, power_db, min_frames, max_frames, n_frames, threshold_pct):
    """Vectorized SSM candidate detection."""
    n_beats = len(beats)
    if n_beats < 4:
        return []
    
    candidates = []
    power = np.max(power_db, axis=0)
    
    # Upper triangle values for threshold
    ssm_upper = np.triu(ssm, k=1)
    valid_vals = ssm_upper[ssm_upper > 0.05]
    if len(valid_vals) < 10:
        return []
    
    threshold = np.percentile(valid_vals, threshold_pct)
    
    # Process diagonals in batches
    max_offset = min(n_beats, 200)
    
    for offset in range(1, max_offset):
        diag = np.diag(ssm, k=offset)
        if len(diag) < 2:
            continue
        
        # Smooth and threshold
        diag_smooth = uniform_filter1d(diag, size=3)
        above_thresh = diag_smooth >= threshold
        
        if not np.any(above_thresh):
            continue
        
        # Find contiguous regions using diff
        changes = np.diff(above_thresh.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        if above_thresh[0]:
            starts = np.concatenate([[0], starts])
        if above_thresh[-1]:
            ends = np.concatenate([ends, [len(above_thresh)]])
        
        for s, e in zip(starts, ends):
            if e - s < 1:
                continue
            
            beat_start = s
            beat_end = s + offset
            
            if beat_end >= n_beats:
                continue
            
            frame_start = beats[beat_start]
            frame_end = beats[beat_end]
            loop_len = frame_end - frame_start
            
            if loop_len < min_frames or loop_len > max_frames:
                continue
            
            sim = float(np.mean(diag_smooth[s:e]))
            ps = power[min(int(frame_start), len(power) - 1)]
            pe = power[max(0, min(int(frame_end) - 1, len(power) - 1))]
            
            candidates.append(LoopPair(
                _loop_start_frame_idx=int(frame_start),
                _loop_end_frame_idx=int(frame_end),
                note_distance=1.0 - sim,
                loudness_difference=abs(ps - pe),
            ))
    
    return candidates


def _downbeat_candidates_fast(feat: Features, min_frames: int, max_frames: int) -> list[LoopPair]:
    """Fast downbeat-aligned candidate generation with proper note_distance."""
    candidates = []
    downbeats = feat.downbeats
    n_db = len(downbeats)
    power = np.max(feat.power_db, axis=0)
    chroma = feat.chroma_cens
    n_f = feat.n_frames
    
    if n_db < 2:
        return []
    
    # All pairs of downbeats within range
    for i in range(n_db):
        for j in range(i + 1, n_db):
            loop_len = downbeats[j] - downbeats[i]
            
            if loop_len < min_frames:
                continue
            if loop_len > max_frames:
                break  # No point checking further for this i
            
            frame_i = min(int(downbeats[i]), n_f - 1)
            frame_j = max(0, min(int(downbeats[j]) - 1, n_f - 1))
            
            # Compute chroma similarity (note_distance)
            c1 = chroma[:, frame_i]
            c2 = chroma[:, frame_j]
            dot = np.dot(c1, c2)
            n1 = np.linalg.norm(c1)
            n2 = np.linalg.norm(c2)
            sim = dot / (n1 * n2 + 1e-10)
            
            ps = power[frame_i]
            pe = power[frame_j]
            
            candidates.append(LoopPair(
                _loop_start_frame_idx=int(downbeats[i]),
                _loop_end_frame_idx=int(downbeats[j]),
                note_distance=1.0 - sim,  # Now computed properly!
                loudness_difference=abs(ps - pe),
            ))
    
    return candidates


def _score_vectorized(
    mlaudio: MLAudio,
    feat: Features,
    candidates: list[LoopPair],
    disable_pruning: bool,
) -> list[LoopPair]:
    """Fully vectorized scoring for maximum performance."""
    if not candidates:
        return []

    # Prune if needed
    if len(candidates) >= 80 and not disable_pruning:
        candidates = _prune_candidates_fast(candidates)

    n = len(candidates)
    starts = np.array([c._loop_start_frame_idx for c in candidates], dtype=np.int64)
    ends = np.array([c._loop_end_frame_idx for c in candidates], dtype=np.int64)

    # Context sizes (adaptive)
    ctx_beat = min(max(10, feat.beat_length), feat.n_frames // 4)
    ctx_bar = min(max(20, feat.bar_length), feat.n_frames // 3)

    # Weights for temporal decay
    w_beat = np.geomspace(4, 1, num=ctx_beat).astype(np.float32)
    w_bar = np.geomspace(3, 1, num=ctx_bar).astype(np.float32)

    # === VECTORIZED SCORING ===
    
    # Melodic/harmonic (numba parallel)
    melody_cqt = _score_features_par(feat.chroma_cqt, starts, ends, ctx_beat, w_beat)
    harmonic = _score_features_par(feat.chroma_cens, starts, ends, ctx_bar, w_bar)
    tonnetz_score = _score_features_par(feat.tonnetz, starts, ends, ctx_bar, w_bar)
    timbral = _score_features_par(feat.mfcc, starts, ends, ctx_bar, w_bar)

    # Point features (fully vectorized)
    spectral = _score_point_match_vec(feat.spectral_centroid, feat.spectral_bandwidth, feat.spectral_flatness, starts, ends)
    energy = _score_energy_vec(feat.rms, starts, ends, feat.n_frames)
    
    # === RHYTHM PROTECTION (CRITICAL!) ===
    # Beat phase alignment - MUST match exactly
    phase = _score_phase_vec(feat.beat_phase, starts, ends, feat.n_frames)
    
    # Sub-beat phase (16th notes) - for tight rhythmic alignment
    sub_phase = _score_sub_phase_vec(feat.sub_beat_phase, starts, ends, feat.n_frames)
    
    # Transient avoidance - don't cut at drum hits!
    transient_avoid = _score_transient_vec(feat.transient_strength, feat.onset_peaks, starts, ends, feat.n_frames)
    
    # Onset envelope continuity - rhythm pattern must flow
    onset_continuity = _score_onset_continuity(feat.onset_env, starts, ends, ctx_beat, feat.n_frames)
    
    # Spectral flux continuity - avoid sudden timbral changes
    flux_continuity = _score_flux_continuity(feat.spectral_flux, starts, ends, ctx_beat, feat.n_frames)
    
    # Phrase alignment
    phrase = _score_phrase_vec(starts, ends, feat.bar_length, feat.beat_length, feat.downbeats)
    
    # Masking (perceptual)
    masking = _score_masking_vec(feat.masking_curve, starts, ends, feat.n_frames)
    
    # Crossfade quality (sample-level waveform correlation)
    crossfade = _score_crossfade_optimized(feat.audio, feat.sr, feat.hop, starts, ends)

    # === RHYTHM GATE: Hard filter for rhythmic alignment ===
    # Candidates with bad rhythm get ZEROED before further scoring
    rhythm_gate = np.ones(n, dtype=np.float64)
    for i in range(n):
        # Phase must be good
        if phase[i] < 0.4:
            rhythm_gate[i] = 0.15  # Very harsh
        elif phase[i] < 0.6:
            rhythm_gate[i] = 0.5
        
        # Sub-phase must be good
        if sub_phase[i] < 0.3:
            rhythm_gate[i] *= 0.3
        
        # No transients allowed
        if transient_avoid[i] < 0.3:
            rhythm_gate[i] *= 0.2
    
    # === COMBINE BASE SCORES ===
    # Weighted combination with MAXIMUM rhythm protection
    base_combined = (
        melody_cqt * 0.06 +        # Melodic content
        harmonic * 0.06 +          # Harmonic content  
        tonnetz_score * 0.03 +     # Tonal space
        timbral * 0.04 +           # Timbre
        spectral * 0.04 +          # Spectral continuity
        energy * 0.05 +            # Energy match
        phase * 0.14 +             # Beat phase - CRITICAL
        sub_phase * 0.12 +         # Sub-beat phase - CRITICAL  
        transient_avoid * 0.18 +   # CRITICAL: avoid drum hits
        onset_continuity * 0.08 +  # Rhythm flow
        flux_continuity * 0.04 +   # Timbral flow
        phrase * 0.05 +            # Phrase structure
        masking * 0.03 +           # Perceptual masking
        crossfade * 0.08           # Crossfade quality
    )
    
    # Apply rhythm gate
    base_combined = base_combined * rhythm_gate

    # === NEURAL ENHANCEMENT (if available) ===
    # Neural only enhances, doesn't override rhythm
    try:
        from pymusiclooper.neural_scoring import apply_neural_enhancement
        neural_scores = apply_neural_enhancement(feat, candidates, base_combined, use_neural=True)
        # Blend: base has MORE weight to preserve rhythm
        combined = base_combined * 0.70 + neural_scores * 0.30
        logging.info("Applied neural enhancement to scoring")
    except Exception as e:
        logging.debug(f"Neural enhancement skipped: {e}")
        combined = base_combined

    # Update candidates
    for i, c in enumerate(candidates):
        c.score = float(combined[i])

    # Filter low scores
    threshold = max(0.3, np.percentile(combined, 20))
    return [c for c in candidates if c.score >= threshold]


@njit(cache=True, parallel=True, fastmath=True)
def _score_features_par(features: np.ndarray, starts: np.ndarray, ends: np.ndarray, ctx: int, weights: np.ndarray) -> np.ndarray:
    """Parallel feature scoring with weighted cosine similarity."""
    n = len(starts)
    n_frames = features.shape[1]
    n_feat = features.shape[0]
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1, b2 = starts[i], ends[i]
        
        # Forward similarity (after loop point)
        fwd = _weighted_cosine_sim(features, b1, b2, ctx, n_frames, n_feat, weights, True)
        
        # Backward similarity (before loop point)
        bwd = _weighted_cosine_sim(features, b1, b2, ctx, n_frames, n_feat, weights, False)
        
        scores[i] = (fwd + bwd) / 2.0
    
    return scores


@njit(cache=True, fastmath=True)
def _weighted_cosine_sim(features: np.ndarray, b1: int, b2: int, ctx: int, n_frames: int, n_feat: int, weights: np.ndarray, forward: bool) -> float:
    """Weighted cosine similarity between two positions."""
    if forward:
        end1 = min(b1 + ctx, n_frames)
        end2 = min(b2 + ctx, n_frames)
        length = min(end1 - b1, end2 - b2)
        s1, s2 = b1, b2
    else:
        length = min(min(ctx, b1), b2)
        s1 = b1 - length
        s2 = b2 - length
    
    if length <= 0:
        return 0.0
    
    total = 0.0
    total_w = 0.0
    
    for j in range(length):
        dot = 0.0
        n1 = 0.0
        n2 = 0.0
        
        for k in range(n_feat):
            v1 = features[k, s1 + j]
            v2 = features[k, s2 + j]
            dot += v1 * v2
            n1 += v1 * v1
            n2 += v2 * v2
        
        denom = np.sqrt(n1 * n2)
        sim = dot / denom if denom > 1e-10 else 0.0
        
        w = weights[j] if j < len(weights) else 1.0
        total += sim * w
        total_w += w
    
    return total / total_w if total_w > 0 else 0.0


@njit(cache=True, parallel=True, fastmath=True)
def _score_point_match_vec(centroid: np.ndarray, bandwidth: np.ndarray, flatness: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Vectorized spectral point matching."""
    n = len(starts)
    n_f = len(centroid)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        c_diff = abs(centroid[b1] - centroid[b2])
        bw_diff = abs(bandwidth[b1] - bandwidth[b2])
        fl_diff = abs(flatness[b1] - flatness[b2])
        
        c_score = max(0.0, 1.0 - c_diff * 5)
        bw_score = max(0.0, 1.0 - bw_diff * 5)
        fl_score = max(0.0, 1.0 - fl_diff * 3)
        
        scores[i] = c_score * 0.4 + bw_score * 0.3 + fl_score * 0.3
    
    return scores


@njit(cache=True, parallel=True, fastmath=True)
def _score_energy_vec(rms: np.ndarray, starts: np.ndarray, ends: np.ndarray, n_frames: int) -> np.ndarray:
    """Vectorized energy matching."""
    n = len(starts)
    n_f = len(rms)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        r1 = rms[b1]
        r2 = rms[b2]
        
        diff = abs(r1 - r2)
        scores[i] = max(0.0, 1.0 - diff * 4)
    
    return scores


@njit(cache=True, parallel=True, fastmath=True)
def _score_phase_vec(beat_phase: np.ndarray, starts: np.ndarray, ends: np.ndarray, n_frames: int) -> np.ndarray:
    """STRICT beat phase scoring - phases MUST align or loop will skip."""
    n = len(starts)
    n_f = len(beat_phase)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        p1 = beat_phase[b1]
        p2 = beat_phase[b2]
        
        # Circular difference
        diff = abs(p1 - p2)
        if diff > 0.5:
            diff = 1.0 - diff
        
        # VERY STRICT: even 10% phase diff is bad
        if diff > 0.15:
            scores[i] = 0.1  # Harsh penalty
        elif diff > 0.08:
            scores[i] = 0.4
        elif diff > 0.04:
            scores[i] = 0.7
        else:
            scores[i] = 1.0 - diff * 8  # Tight tolerance
        
        # Bonus for being exactly on beat (phase near 0 or 1)
        on_beat_1 = p1 < 0.06 or p1 > 0.94
        on_beat_2 = p2 < 0.06 or p2 > 0.94
        if on_beat_1 and on_beat_2:
            scores[i] = min(1.0, scores[i] + 0.15)
        
        scores[i] = max(0.0, min(1.0, scores[i]))
    
    return scores


@njit(cache=True, parallel=True, fastmath=True)
def _score_sub_phase_vec(sub_phase: np.ndarray, starts: np.ndarray, ends: np.ndarray, n_frames: int) -> np.ndarray:
    """STRICT sub-beat (16th note) phase scoring for tight rhythmic cuts."""
    n = len(starts)
    n_f = len(sub_phase)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        p1 = sub_phase[b1]
        p2 = sub_phase[b2]
        
        diff = abs(p1 - p2)
        if diff > 0.5:
            diff = 1.0 - diff
        
        # Stricter penalties for micro-timing
        if diff > 0.12:
            scores[i] = 0.15  # Very bad
        elif diff > 0.06:
            scores[i] = 0.5
        else:
            scores[i] = 1.0 - diff * 10
        
        scores[i] = max(0.0, min(1.0, scores[i]))
    
    return scores


@njit(cache=True, parallel=True, fastmath=True)
def _score_transient_vec(transient: np.ndarray, onset_peaks: np.ndarray, starts: np.ndarray, ends: np.ndarray, n_frames: int) -> np.ndarray:
    """CRITICAL: Avoid cutting at drum hits / transients. Strong penalty for bad cuts."""
    n = len(starts)
    n_f = len(transient)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        # Transient strength at exact cut points (low = good)
        t1 = transient[b1]
        t2 = transient[b2]
        
        # HARSH penalty for cutting at transients
        if t1 > 0.7 or t2 > 0.7:
            scores[i] = 0.1  # Very bad
            continue
        if t1 > 0.5 or t2 > 0.5:
            scores[i] = 0.3  # Bad
            continue
        
        transient_score = 1.0 - (t1 + t2)  # More aggressive
        
        # Onset peaks (binary) - NEVER cut at onset peak
        o1 = onset_peaks[b1]
        o2 = onset_peaks[b2]
        if o1 > 0.5 or o2 > 0.5:
            scores[i] = 0.2  # Very bad to cut at onset
            continue
        
        # Check window around cut points (±5 frames)
        window = 5
        max_trans_start = 0.0
        max_trans_end = 0.0
        onset_near_start = 0.0
        onset_near_end = 0.0
        
        for j in range(max(0, b1 - window), min(n_f, b1 + window + 1)):
            if transient[j] > max_trans_start:
                max_trans_start = transient[j]
            if onset_peaks[j] > onset_near_start:
                onset_near_start = onset_peaks[j]
                
        for j in range(max(0, b2 - window), min(n_f, b2 + window + 1)):
            if transient[j] > max_trans_end:
                max_trans_end = transient[j]
            if onset_peaks[j] > onset_near_end:
                onset_near_end = onset_peaks[j]
        
        # Penalty for transients near the cut
        window_trans = 1.0 - (max_trans_start + max_trans_end) / 2 * 0.5
        
        # Penalty for onsets near the cut
        window_onset = 1.0 - (onset_near_start + onset_near_end) / 2 * 0.3
        
        scores[i] = transient_score * 0.4 + window_trans * 0.35 + window_onset * 0.25
        scores[i] = max(0.0, min(1.0, scores[i]))
    
    return scores


def _score_onset_continuity(onset_env: np.ndarray, starts: np.ndarray, ends: np.ndarray, ctx: int, n_frames: int) -> np.ndarray:
    """Score rhythm continuity - onset envelope should flow smoothly across the cut."""
    n = len(starts)
    scores = np.zeros(n, dtype=np.float64)
    n_f = len(onset_env)
    
    for i in range(n):
        b1 = int(starts[i])
        b2 = int(ends[i])
        
        # Get onset envelope before end and after start
        pre_start = max(0, b2 - ctx)
        pre_end = min(b2, n_f)
        post_start = max(0, b1)
        post_end = min(b1 + ctx, n_f)
        
        pre_len = pre_end - pre_start
        post_len = post_end - post_start
        min_len = min(pre_len, post_len)
        
        if min_len < 4:
            scores[i] = 0.5
            continue
        
        # Compare onset patterns
        pre = onset_env[pre_end - min_len:pre_end]
        post = onset_env[post_start:post_start + min_len]
        
        # Correlation of onset patterns
        if np.std(pre) > 1e-8 and np.std(post) > 1e-8:
            corr = np.corrcoef(pre, post)[0, 1]
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.0
        
        # Point match at the cut
        o1 = onset_env[min(b1, n_f - 1)]
        o2 = onset_env[max(0, min(b2 - 1, n_f - 1))]
        point_match = 1.0 - abs(o1 - o2)
        
        scores[i] = (corr + 1) / 2 * 0.6 + point_match * 0.4
    
    return scores


def _score_flux_continuity(spectral_flux: np.ndarray, starts: np.ndarray, ends: np.ndarray, ctx: int, n_frames: int) -> np.ndarray:
    """Score spectral flux continuity - timbral changes should flow smoothly."""
    n = len(starts)
    scores = np.zeros(n, dtype=np.float64)
    n_f = len(spectral_flux)
    
    for i in range(n):
        b1 = int(starts[i])
        b2 = int(ends[i])
        
        # Flux at cut points (low flux = stable timbre = good for cutting)
        f1 = spectral_flux[min(b1, n_f - 1)]
        f2 = spectral_flux[max(0, min(b2 - 1, n_f - 1))]
        
        # Low flux at cut points is good
        flux_low = 1.0 - (f1 + f2) / 2
        
        # Flux should match (similar timbral activity)
        flux_match = 1.0 - abs(f1 - f2) * 2
        flux_match = max(0, flux_match)
        
        # Check window around cut for stability
        window = min(5, ctx // 4)
        pre_flux = spectral_flux[max(0, b2 - window):min(b2, n_f)]
        post_flux = spectral_flux[max(0, b1):min(b1 + window, n_f)]
        
        if len(pre_flux) > 0 and len(post_flux) > 0:
            pre_mean = np.mean(pre_flux)
            post_mean = np.mean(post_flux)
            window_match = 1.0 - abs(pre_mean - post_mean) * 2
            window_match = max(0, window_match)
        else:
            window_match = 0.5
        
        scores[i] = flux_low * 0.3 + flux_match * 0.4 + window_match * 0.3
    
    return scores


@njit(cache=True, parallel=True, fastmath=True)
def _score_masking_vec(masking: np.ndarray, starts: np.ndarray, ends: np.ndarray, n_frames: int) -> np.ndarray:
    """Vectorized perceptual masking scoring."""
    n = len(starts)
    n_f = len(masking)
    scores = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        b1 = min(starts[i], n_f - 1)
        b2 = max(0, min(ends[i] - 1, n_f - 1))
        
        # Higher masking = transition less audible = better
        m1 = masking[b1]
        m2 = masking[b2]
        
        scores[i] = (m1 + m2) / 2
    
    return scores


def _score_phrase_vec(starts: np.ndarray, ends: np.ndarray, bar_length: int, beat_length: int, downbeats: np.ndarray) -> np.ndarray:
    """Vectorized phrase structure scoring."""
    n = len(starts)
    scores = np.zeros(n, dtype=np.float64)
    
    if bar_length <= 0 or beat_length <= 0:
        return scores + 0.5
    
    # Ideal bar counts
    ideal_bars = np.array([4, 8, 16, 32])
    
    for i in range(n):
        loop_len = ends[i] - starts[i]
        
        # Bar alignment score
        bars = loop_len / bar_length
        len_score = 0.0
        for target in ideal_bars:
            for mult in [0.5, 1.0, 2.0]:
                deviation = abs(bars - target * mult) / (target * mult * 0.2 + 0.01)
                close = max(0, 1.0 - deviation)
                len_score = max(len_score, close)
        
        # Beat alignment
        start_off = starts[i] % beat_length
        end_off = ends[i] % beat_length
        beat_align = 1.0 - (start_off + end_off) / (2 * beat_length + 0.01)
        
        # Downbeat proximity
        down_align = 0.0
        if len(downbeats) > 0:
            s_dist = np.min(np.abs(downbeats - starts[i]))
            e_dist = np.min(np.abs(downbeats - ends[i]))
            max_dist = bar_length
            down_align = 1.0 - min(s_dist + e_dist, 2 * max_dist) / (2 * max_dist + 0.01)
        
        scores[i] = len_score * 0.4 + beat_align * 0.3 + down_align * 0.3
    
    return scores


def _score_crossfade_optimized(audio: np.ndarray, sr: int, hop: int, starts: np.ndarray, ends: np.ndarray, fade_ms: int = 50) -> np.ndarray:
    """Crossfade quality scoring with waveform analysis for seamless transitions."""
    n = len(starts)
    scores = np.zeros(n, dtype=np.float64)
    fade_samples = int(sr * fade_ms / 1000)
    audio_len = len(audio)
    
    for i in range(n):
        b1_samples = int(starts[i]) * hop
        b2_samples = int(ends[i]) * hop
        
        pre_end = b2_samples
        pre_start = max(0, pre_end - fade_samples)
        post_start = b1_samples
        post_end = min(audio_len, post_start + fade_samples)
        
        fade_len = min(pre_end - pre_start, post_end - post_start)
        
        if fade_len < 32:
            scores[i] = 0.3
            continue
        
        pre = audio[pre_end - fade_len:pre_end]
        post = audio[post_start:post_start + fade_len]
        
        # 1. Waveform correlation at transition
        std_pre = np.std(pre)
        std_post = np.std(post)
        
        if std_pre > 1e-8 and std_post > 1e-8:
            corr = np.corrcoef(pre, post)[0, 1]
            corr = 0.0 if np.isnan(corr) else max(0, corr)
        else:
            corr = 0.3
        
        # 2. Energy envelope match
        # Split into small windows and compare energy profile
        n_windows = min(8, fade_len // 64)
        if n_windows >= 2:
            win_size = fade_len // n_windows
            pre_env = np.array([np.mean(pre[j*win_size:(j+1)*win_size]**2) for j in range(n_windows)])
            post_env = np.array([np.mean(post[j*win_size:(j+1)*win_size]**2) for j in range(n_windows)])
            
            if np.std(pre_env) > 1e-10 and np.std(post_env) > 1e-10:
                env_corr = np.corrcoef(pre_env, post_env)[0, 1]
                env_corr = 0.0 if np.isnan(env_corr) else max(0, env_corr)
            else:
                env_corr = 0.5
        else:
            e_pre = np.mean(pre ** 2)
            e_post = np.mean(post ** 2)
            e_diff = abs(e_pre - e_post) / (max(e_pre, e_post) + 1e-10)
            env_corr = 1.0 - min(e_diff, 1.0)
        
        # 3. Zero-crossing rate match (rhythm/texture)
        zcr_pre = np.sum(np.abs(np.diff(np.sign(pre)))) / (2 * fade_len + 1e-10)
        zcr_post = np.sum(np.abs(np.diff(np.sign(post)))) / (2 * fade_len + 1e-10)
        zcr_match = 1.0 - min(abs(zcr_pre - zcr_post) * 3, 1.0)
        
        # 4. Amplitude at cut point should be low (near zero)
        cut_amp_pre = abs(pre[-1]) / (np.max(np.abs(pre)) + 1e-10)
        cut_amp_post = abs(post[0]) / (np.max(np.abs(post)) + 1e-10)
        low_amp = 1.0 - (cut_amp_pre + cut_amp_post) / 2
        
        # 5. Smoothness of simulated crossfade
        fade_out = np.linspace(1, 0, min(fade_len, 256))
        fade_in = np.linspace(0, 1, min(fade_len, 256))
        mixed = pre[-len(fade_out):] * fade_out + post[:len(fade_in)] * fade_in
        smoothness = 1.0 / (1.0 + np.std(np.diff(mixed)) * 20)
        
        scores[i] = corr * 0.25 + env_corr * 0.25 + zcr_match * 0.2 + low_amp * 0.15 + smoothness * 0.15
    
    return scores


def _prune_candidates_fast(candidates: list[LoopPair]) -> list[LoopPair]:
    """Fast candidate pruning."""
    if len(candidates) <= 100:
        return candidates
    
    # Sort by note_distance and loudness
    candidates.sort(key=lambda x: (x.note_distance, x.loudness_difference))
    
    # Keep top candidates
    n_keep = min(len(candidates), max(100, len(candidates) // 4))
    return candidates[:n_keep]


def _optimize_selection(scored: list[LoopPair], feat: Features) -> None:
    """Optimize selection for diversity and quality. Score stays in [0, 1]."""
    if len(scored) <= 3:
        return
    
    bar_len = feat.bar_length
    beat_len = feat.beat_length
    
    for p in scored:
        dur = p._loop_end_frame_idx - p._loop_start_frame_idx
        
        # Small bonus for ideal bar lengths (max 0.05)
        bar_bonus = 0.0
        if bar_len > 0:
            bars = dur / bar_len
            for target in [4, 8, 16, 32]:
                if abs(bars - target) < 0.5:
                    bar_bonus = (1.0 - abs(bars - target) * 2) * 0.05
                    break
        
        # Small bonus for beat alignment (max 0.03)
        beat_bonus = 0.0
        if beat_len > 0:
            start_on = p._loop_start_frame_idx % beat_len < beat_len * 0.15
            end_on = p._loop_end_frame_idx % beat_len < beat_len * 0.15
            beat_bonus = (float(start_on) * 0.5 + float(end_on) * 0.5) * 0.03
        
        # Apply small bonuses, keep score in [0, 1]
        p.score = min(1.0, p.score + bar_bonus + beat_bonus)


def _finalize_with_alignment(mlaudio: MLAudio, feat: Features, scored: list[LoopPair]) -> None:
    """Finalize loop points with sample-accurate zero-crossing alignment."""
    for pair in scored:
        # Convert frames to samples
        start_samples = mlaudio.frames_to_samples(mlaudio.apply_trim_offset(pair._loop_start_frame_idx))
        end_samples = mlaudio.frames_to_samples(mlaudio.apply_trim_offset(pair._loop_end_frame_idx))
        
        # Align to zero-crossings for click-free transitions
        pair.loop_start = _find_optimal_zero_crossing(mlaudio.playback_audio, mlaudio.rate, start_samples)
        pair.loop_end = _find_optimal_zero_crossing(mlaudio.playback_audio, mlaudio.rate, end_samples)


def _find_optimal_zero_crossing(audio: np.ndarray, sr: int, target_sample: int) -> int:
    """Find optimal zero-crossing near target for seamless loop."""
    search_ms = 5
    search_samples = int(sr * search_ms / 1000)
    
    start = max(0, target_sample - search_samples)
    end = min(len(audio), target_sample + search_samples)
    
    if end <= start:
        return max(0, min(target_sample, len(audio) - 1))
    
    # Get mono signal for zero-crossing detection
    if len(audio.shape) == 2:
        mono = np.mean(audio[start:end], axis=1)
    else:
        mono = audio[start:end]
    
    if len(mono) < 2:
        return target_sample
    
    # Find zero crossings
    signs = np.sign(mono)
    crossings = np.where(np.diff(signs) != 0)[0]
    
    if len(crossings) == 0:
        return target_sample
    
    # Find crossing closest to target
    relative_target = target_sample - start
    closest_idx = crossings[np.argmin(np.abs(crossings - relative_target))]
    
    return start + closest_idx


# Alias for backwards compatibility
def nearest_zero_crossing(audio: np.ndarray, sr: int, target: int) -> int:
    """Find nearest zero crossing (backwards compatible)."""
    return _find_optimal_zero_crossing(audio, sr, target)
