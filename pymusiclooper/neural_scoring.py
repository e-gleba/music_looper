"""
Premium Neural Loop Scoring with Expert Ensemble + Advanced Algorithms.

Expert Network Architecture:
- HarmonicExpert: Tonal/harmonic transitions (circle of fifths, key detection)
- RhythmExpert: Beat/timing precision (phase, groove)
- TimbreExpert: Spectral/timbral continuity (formants, brightness)
- DynamicsExpert: Energy/loudness flow (compression, envelope)
- NeuroacousticExpert: Human auditory pathway (masking, roughness)
- OnsetExpert: Attack/transient handling (percussive continuity)
- SpectralFlowExpert: Frequency evolution (spectral flux coherence)
- PhraseExpert: Musical phrase structure (motifs, sections)

Plus:
- Psychoacoustic model (Plomp-Levelt, ERB, temporal masking)
- Neuroacoustic model (consonance, roughness, tension)
- Fourier phase coherence
- Autocorrelation periodicity
- Meta-learner for expert weighting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from pymusiclooper.analysis import Features


# ============================================================================
# PSYCHOACOUSTIC & NEUROACOUSTIC CONSTANTS
# ============================================================================

BARK_BANDS = np.array([
    20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
    1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
    5300, 6400, 7700, 9500, 12000, 15500
], dtype=np.float32)

FORWARD_MASKING_MS = 200
BACKWARD_MASKING_MS = 50
JND_LOUDNESS = 0.5
JND_PITCH = 0.3
JND_TIMING = 10

HARMONIC_RATIOS = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
PERFECT_INTERVALS = np.array([0, 5, 7, 12], dtype=np.int32)
ERB_CONST = 24.7
ERB_FACTOR = 9.265


# ============================================================================
# ADVANCED FEATURE VECTOR
# ============================================================================

@dataclass
class TransitionFeatures:
    """Rich 120-dim feature vector for a transition point."""
    # Chromatic (12 dims each side)
    chroma_pre: np.ndarray
    chroma_post: np.ndarray
    
    # Timbral (20 dims each side) 
    mfcc_pre: np.ndarray
    mfcc_post: np.ndarray
    
    # Harmonic (6 dims each side)
    tonnetz_pre: np.ndarray
    tonnetz_post: np.ndarray
    
    # Dynamics
    rms_pre: float
    rms_post: float
    rms_diff: float
    rms_ratio: float
    
    # Spectral
    centroid_pre: float
    centroid_post: float
    centroid_diff: float
    flatness_pre: float
    flatness_post: float
    bandwidth_pre: float
    bandwidth_post: float
    flux_pre: float
    flux_post: float
    flux_diff: float
    
    # Rhythm
    beat_phase_pre: float
    beat_phase_post: float
    phase_diff: float
    sub_phase_pre: float
    sub_phase_post: float
    sub_phase_diff: float
    transient_pre: float
    transient_post: float
    onset_pre: float
    onset_post: float
    
    # Context
    masking_pre: float
    masking_post: float
    loop_duration: float
    
    # Position
    start_position: float
    end_position: float
    covers_full: float
    
    # Neuroacoustic
    consonance_pre: float
    consonance_post: float
    consonance_diff: float
    roughness_pre: float
    roughness_post: float
    harmonic_tension: float
    
    # Advanced
    phase_coherence: float
    autocorr_score: float
    key_match: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat 120-dim feature vector."""
        return np.concatenate([
            self.chroma_pre,       # 12
            self.chroma_post,      # 12
            self.mfcc_pre,         # 20
            self.mfcc_post,        # 20
            self.tonnetz_pre,      # 6
            self.tonnetz_post,     # 6
            np.array([
                self.rms_pre, self.rms_post, self.rms_diff, self.rms_ratio,
                self.centroid_pre, self.centroid_post, self.centroid_diff,
                self.flatness_pre, self.flatness_post,
                self.bandwidth_pre, self.bandwidth_post,
                self.flux_pre, self.flux_post, self.flux_diff,
                self.beat_phase_pre, self.beat_phase_post, self.phase_diff,
                self.sub_phase_pre, self.sub_phase_post, self.sub_phase_diff,
                self.transient_pre, self.transient_post,
                self.onset_pre, self.onset_post,
                self.masking_pre, self.masking_post,
                self.loop_duration,
                self.start_position, self.end_position, self.covers_full,
                self.consonance_pre, self.consonance_post, self.consonance_diff,
                self.roughness_pre, self.roughness_post,
                self.harmonic_tension,
                self.phase_coherence, self.autocorr_score, self.key_match,
            ])  # 39
        ]).astype(np.float32)
    
    @staticmethod
    def dim() -> int:
        return 12 + 12 + 20 + 20 + 6 + 6 + 39  # = 115


# ============================================================================
# NEUROACOUSTIC MODELS
# ============================================================================

@njit(cache=True, fastmath=True)
def roughness_model(chroma: np.ndarray) -> float:
    """Roughness based on Plomp-Levelt (1965) - dissonance from beating."""
    roughness = 0.0
    for i in range(11):
        r = chroma[i] * chroma[i + 1] * 0.8
        roughness += r
    for i in range(6):
        r = chroma[i] * chroma[i + 6] * 0.3
        roughness += r
    return min(1.0, roughness)


@njit(cache=True, fastmath=True)
def consonance_model(chroma: np.ndarray) -> float:
    """Consonance based on harmonic series and Pythagorean ratios."""
    consonance = 0.0
    total_weight = 0.0
    
    for i in range(12):
        consonance += chroma[i] * chroma[i] * 1.0
        total_weight += chroma[i] * 1.0
    
    for i in range(12):
        j = (i + 7) % 12
        consonance += chroma[i] * chroma[j] * 0.9
        total_weight += (chroma[i] + chroma[j]) * 0.45
    
    for i in range(12):
        j = (i + 5) % 12
        consonance += chroma[i] * chroma[j] * 0.85
        total_weight += (chroma[i] + chroma[j]) * 0.425
    
    for i in range(12):
        j = (i + 4) % 12
        consonance += chroma[i] * chroma[j] * 0.75
        total_weight += (chroma[i] + chroma[j]) * 0.375
    
    for i in range(12):
        j = (i + 3) % 12
        consonance += chroma[i] * chroma[j] * 0.7
        total_weight += (chroma[i] + chroma[j]) * 0.35
    
    if total_weight < 1e-6:
        return 0.5
    
    return min(1.0, consonance / total_weight)


@njit(cache=True, fastmath=True)
def harmonic_tension_between(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """Harmonic tension based on circle of fifths distance."""
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
    
    interval = abs(max1_idx - max2_idx)
    if interval > 6:
        interval = 12 - interval
    
    tension_map = np.array([0.0, 0.9, 0.6, 0.3, 0.25, 0.1, 0.85])
    base_tension = tension_map[interval]
    
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


@njit(cache=True, fastmath=True)
def key_detection(chroma: np.ndarray) -> tuple:
    """Detect key using Krumhansl-Kessler profiles."""
    # Major key profile
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    # Minor key profile  
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    best_key = 0
    best_mode = 0  # 0 = major, 1 = minor
    best_corr = -2.0
    
    for key in range(12):
        # Rotate chroma to key
        rotated = np.zeros(12)
        for i in range(12):
            rotated[i] = chroma[(i + key) % 12]
        
        # Correlate with major
        maj_corr = 0.0
        min_corr = 0.0
        for i in range(12):
            maj_corr += rotated[i] * major_profile[i]
            min_corr += rotated[i] * minor_profile[i]
        
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = key
            best_mode = 0
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = key
            best_mode = 1
    
    return best_key, best_mode, best_corr


@njit(cache=True, fastmath=True)
def key_distance(key1: int, mode1: int, key2: int, mode2: int) -> float:
    """Distance between keys on circle of fifths (0 = same, 1 = far)."""
    # Circle of fifths distance
    cof_dist = min(abs(key1 - key2), 12 - abs(key1 - key2))
    
    # Mode penalty
    mode_penalty = 0.0 if mode1 == mode2 else 0.15
    
    # Relative major/minor bonus
    if mode1 != mode2:
        rel_key = (key1 + 3) % 12 if mode1 == 0 else (key1 - 3 + 12) % 12
        if rel_key == key2:
            mode_penalty = 0.05  # Relative key is close
    
    return min(1.0, cof_dist / 6.0 + mode_penalty)


@njit(cache=True, fastmath=True)
def phase_coherence_score(phase1: float, phase2: float, sub1: float, sub2: float) -> float:
    """Phase coherence using Fourier-inspired approach."""
    # Convert phases to complex unit vectors
    re1 = np.cos(phase1 * 2 * np.pi)
    im1 = np.sin(phase1 * 2 * np.pi)
    re2 = np.cos(phase2 * 2 * np.pi)
    im2 = np.sin(phase2 * 2 * np.pi)
    
    # Dot product of unit vectors = cos(angle between)
    beat_coherence = re1 * re2 + im1 * im2
    
    # Same for sub-beat
    re1s = np.cos(sub1 * 2 * np.pi)
    im1s = np.sin(sub1 * 2 * np.pi)
    re2s = np.cos(sub2 * 2 * np.pi)
    im2s = np.sin(sub2 * 2 * np.pi)
    
    sub_coherence = re1s * re2s + im1s * im2s
    
    # Combined (beat more important)
    return (beat_coherence * 0.65 + sub_coherence * 0.35 + 1.0) / 2.0


# ============================================================================
# EXPERT NEURAL NETWORKS
# ============================================================================

class ExpertMLP:
    """Expert network with residual connections and dropout-like regularization."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 24, name: str = "expert"):
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / (hidden_dim // 2))
        
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim // 2).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim // 2, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim // 2, 1).astype(np.float32) * scale3
        self.b3 = np.zeros(1, dtype=np.float32)
        self.W_res = np.random.randn(input_dim, hidden_dim // 2).astype(np.float32) * scale1 * 0.5
    
    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        h1_raw = X @ self.W1 + self.b1
        h1 = np.where(h1_raw > 0, h1_raw, 0.01 * h1_raw)
        h2_raw = h1 @ self.W2 + self.b2
        h2 = np.where(h2_raw > 0, h2_raw, 0.01 * h2_raw)
        res = X @ self.W_res
        h2 = h2 + res * 0.3
        out = h2 @ self.W3 + self.b3
        return 1.0 / (1.0 + np.exp(-np.clip(out, -15, 15)))
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 80, lr: float = 0.012):
        n = len(X)
        mW1 = np.zeros_like(self.W1)
        mW2 = np.zeros_like(self.W2)
        mW3 = np.zeros_like(self.W3)
        mb1 = np.zeros_like(self.b1)
        mb2 = np.zeros_like(self.b2)
        mb3 = np.zeros_like(self.b3)
        mW_res = np.zeros_like(self.W_res)
        beta = 0.9
        
        for epoch in range(epochs):
            h1_raw = X @ self.W1 + self.b1
            h1 = np.where(h1_raw > 0, h1_raw, 0.01 * h1_raw)
            h2_raw = h1 @ self.W2 + self.b2
            h2_base = np.where(h2_raw > 0, h2_raw, 0.01 * h2_raw)
            res = X @ self.W_res
            h2 = h2_base + res * 0.3
            out = h2 @ self.W3 + self.b3
            pred = 1.0 / (1.0 + np.exp(-np.clip(out, -15, 15)))
            
            d_out = (pred.flatten() - y).reshape(-1, 1) * pred * (1 - pred) / n
            d_W3 = h2.T @ d_out
            d_b3 = np.sum(d_out, axis=0)
            d_h2 = d_out @ self.W3.T
            d_W_res = X.T @ (d_h2 * 0.3)
            d_h2_base = d_h2 * np.where(h2_raw > 0, 1, 0.01)
            d_W2 = h1.T @ d_h2_base
            d_b2 = np.sum(d_h2_base, axis=0)
            d_h1 = d_h2_base @ self.W2.T
            d_h1 = d_h1 * np.where(h1_raw > 0, 1, 0.01)
            d_W1 = X.T @ d_h1
            d_b1 = np.sum(d_h1, axis=0)
            
            mW1 = beta * mW1 + (1 - beta) * d_W1
            mW2 = beta * mW2 + (1 - beta) * d_W2
            mW3 = beta * mW3 + (1 - beta) * d_W3
            mb1 = beta * mb1 + (1 - beta) * d_b1
            mb2 = beta * mb2 + (1 - beta) * d_b2
            mb3 = beta * mb3 + (1 - beta) * d_b3
            mW_res = beta * mW_res + (1 - beta) * d_W_res
            
            self.W1 -= lr * mW1
            self.W2 -= lr * mW2
            self.W3 -= lr * mW3
            self.b1 -= lr * mb1
            self.b2 -= lr * mb2
            self.b3 -= lr * mb3
            self.W_res -= lr * mW_res


class MetaLearner:
    """Meta-learner that learns optimal expert weights for this composition."""
    
    def __init__(self, n_experts: int = 8, input_dim: int = 115):
        self.n_experts = n_experts
        scale = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim, n_experts).astype(np.float32) * scale
        self.b = np.zeros(n_experts, dtype=np.float32)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return softmax weights for each expert per sample."""
        logits = X @ self.W + self.b
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def train(self, X: np.ndarray, expert_scores: np.ndarray, targets: np.ndarray, epochs: int = 50, lr: float = 0.02):
        """Train meta-learner to weight experts optimally."""
        n = len(X)
        mW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b)
        beta = 0.9
        
        for epoch in range(epochs):
            weights = self.forward(X)  # [n, n_experts]
            
            # Weighted sum of expert scores
            pred = np.sum(weights * expert_scores, axis=1)  # [n]
            
            # MSE loss gradient
            error = pred - targets  # [n]
            
            # Gradient through softmax-weighted sum
            d_weights = error.reshape(-1, 1) * expert_scores / n
            
            # Softmax gradient
            d_logits = weights * (d_weights - np.sum(d_weights * weights, axis=1, keepdims=True))
            
            d_W = X.T @ d_logits
            d_b = np.sum(d_logits, axis=0)
            
            mW = beta * mW + (1 - beta) * d_W
            mb = beta * mb + (1 - beta) * d_b
            
            self.W -= lr * mW
            self.b -= lr * mb


class ExpertEnsemble:
    """8-Expert Ensemble with Meta-Learner."""
    
    def __init__(self, input_dim: int = 115):
        # Specialized experts
        self.harmonic_expert = ExpertMLP(input_dim, hidden_dim=32, name="harmonic")
        self.rhythm_expert = ExpertMLP(input_dim, hidden_dim=40, name="rhythm")
        self.timbre_expert = ExpertMLP(input_dim, hidden_dim=32, name="timbre")
        self.dynamics_expert = ExpertMLP(input_dim, hidden_dim=28, name="dynamics")
        self.neuroacoustic_expert = ExpertMLP(input_dim, hidden_dim=36, name="neuroacoustic")
        self.onset_expert = ExpertMLP(input_dim, hidden_dim=32, name="onset")
        self.spectral_flow_expert = ExpertMLP(input_dim, hidden_dim=28, name="spectral_flow")
        self.phrase_expert = ExpertMLP(input_dim, hidden_dim=24, name="phrase")
        
        self.meta_learner = MetaLearner(n_experts=8, input_dim=input_dim)
        self.trained = False
        self.X_mean = None
        self.X_std = None
        
        # Default weights (before meta-learning)
        self.default_weights = np.array([
            0.12,  # harmonic
            0.22,  # rhythm - most important
            0.12,  # timbre
            0.10,  # dynamics
            0.16,  # neuroacoustic
            0.12,  # onset
            0.08,  # spectral_flow
            0.08,  # phrase
        ], dtype=np.float32)
    
    def forward_batch(self, X: np.ndarray, use_meta: bool = True) -> np.ndarray:
        """Batch prediction with all experts."""
        h = self.harmonic_expert.forward_batch(X).flatten()
        r = self.rhythm_expert.forward_batch(X).flatten()
        t = self.timbre_expert.forward_batch(X).flatten()
        d = self.dynamics_expert.forward_batch(X).flatten()
        n = self.neuroacoustic_expert.forward_batch(X).flatten()
        o = self.onset_expert.forward_batch(X).flatten()
        sf = self.spectral_flow_expert.forward_batch(X).flatten()
        p = self.phrase_expert.forward_batch(X).flatten()
        
        expert_scores = np.stack([h, r, t, d, n, o, sf, p], axis=1)  # [n, 8]
        
        if use_meta and self.trained:
            weights = self.meta_learner.forward(X)
            return np.sum(weights * expert_scores, axis=1)
        else:
            return np.sum(self.default_weights * expert_scores, axis=1)
    
    def train_all(self, X: np.ndarray, 
                  y_harmonic: np.ndarray, y_rhythm: np.ndarray,
                  y_timbre: np.ndarray, y_dynamics: np.ndarray,
                  y_neuroacoustic: np.ndarray, y_onset: np.ndarray,
                  y_spectral_flow: np.ndarray, y_phrase: np.ndarray,
                  y_combined: np.ndarray,
                  epochs: int = 70):
        """Train all experts with specialized targets."""
        logging.debug("Training 8 expert networks...")
        
        self.harmonic_expert.train(X, y_harmonic, epochs=epochs)
        self.rhythm_expert.train(X, y_rhythm, epochs=epochs + 30)  # Extra epochs
        self.timbre_expert.train(X, y_timbre, epochs=epochs)
        self.dynamics_expert.train(X, y_dynamics, epochs=epochs)
        self.neuroacoustic_expert.train(X, y_neuroacoustic, epochs=epochs + 15)
        self.onset_expert.train(X, y_onset, epochs=epochs + 20)
        self.spectral_flow_expert.train(X, y_spectral_flow, epochs=epochs)
        self.phrase_expert.train(X, y_phrase, epochs=epochs)
        
        # Train meta-learner on expert outputs
        logging.debug("Training meta-learner...")
        h = self.harmonic_expert.forward_batch(X).flatten()
        r = self.rhythm_expert.forward_batch(X).flatten()
        t = self.timbre_expert.forward_batch(X).flatten()
        d = self.dynamics_expert.forward_batch(X).flatten()
        n = self.neuroacoustic_expert.forward_batch(X).flatten()
        o = self.onset_expert.forward_batch(X).flatten()
        sf = self.spectral_flow_expert.forward_batch(X).flatten()
        p = self.phrase_expert.forward_batch(X).flatten()
        
        expert_scores = np.stack([h, r, t, d, n, o, sf, p], axis=1)
        self.meta_learner.train(X, expert_scores, y_combined, epochs=60)
        
        self.trained = True


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_transition_features(
    feat: Features,
    start_frame: int,
    end_frame: int,
    context: int = 12,
) -> TransitionFeatures:
    """Extract rich 115-dim features at a transition point."""
    n_f = feat.n_frames
    
    b1 = min(max(0, start_frame), n_f - 1)
    b2 = min(max(0, end_frame - 1), n_f - 1)
    ctx = min(context, b1, n_f - b2 - 1, 25)
    
    # Chroma
    if ctx > 1:
        chroma_pre = np.mean(feat.chroma_cens[:, max(0, b2 - ctx):b2 + 1], axis=1)
        chroma_post = np.mean(feat.chroma_cens[:, b1:min(b1 + ctx + 1, n_f)], axis=1)
    else:
        chroma_pre = feat.chroma_cens[:, b2]
        chroma_post = feat.chroma_cens[:, b1]
    
    # MFCC
    if ctx > 1:
        mfcc_pre = np.mean(feat.mfcc[:, max(0, b2 - ctx):b2 + 1], axis=1)
        mfcc_post = np.mean(feat.mfcc[:, b1:min(b1 + ctx + 1, n_f)], axis=1)
    else:
        mfcc_pre = feat.mfcc[:, b2]
        mfcc_post = feat.mfcc[:, b1]
    
    # Tonnetz
    if ctx > 1:
        tonnetz_pre = np.mean(feat.tonnetz[:, max(0, b2 - ctx):b2 + 1], axis=1)
        tonnetz_post = np.mean(feat.tonnetz[:, b1:min(b1 + ctx + 1, n_f)], axis=1)
    else:
        tonnetz_pre = feat.tonnetz[:, b2]
        tonnetz_post = feat.tonnetz[:, b1]
    
    # Dynamics
    rms_pre = feat.rms[b2]
    rms_post = feat.rms[b1]
    rms_diff = abs(rms_pre - rms_post)
    rms_ratio = min(rms_pre, rms_post) / (max(rms_pre, rms_post) + 1e-10)
    
    # Spectral
    centroid_pre = feat.spectral_centroid[b2]
    centroid_post = feat.spectral_centroid[b1]
    centroid_diff = abs(centroid_pre - centroid_post)
    flatness_pre = feat.spectral_flatness[b2]
    flatness_post = feat.spectral_flatness[b1]
    bandwidth_pre = feat.spectral_bandwidth[b2]
    bandwidth_post = feat.spectral_bandwidth[b1]
    flux_pre = feat.spectral_flux[b2]
    flux_post = feat.spectral_flux[b1]
    flux_diff = abs(flux_pre - flux_post)
    
    # Rhythm
    beat_phase_pre = feat.beat_phase[b2]
    beat_phase_post = feat.beat_phase[b1]
    phase_diff = abs(beat_phase_pre - beat_phase_post)
    if phase_diff > 0.5:
        phase_diff = 1.0 - phase_diff
    
    sub_phase_pre = feat.sub_beat_phase[b2]
    sub_phase_post = feat.sub_beat_phase[b1]
    sub_phase_diff = abs(sub_phase_pre - sub_phase_post)
    if sub_phase_diff > 0.5:
        sub_phase_diff = 1.0 - sub_phase_diff
    
    transient_pre = feat.transient_strength[b2]
    transient_post = feat.transient_strength[b1]
    onset_pre = feat.onset_peaks[b2]
    onset_post = feat.onset_peaks[b1]
    
    # Context
    masking_pre = feat.masking_curve[b2]
    masking_post = feat.masking_curve[b1]
    loop_duration = (end_frame - start_frame) / n_f
    
    # Position
    start_position = start_frame / n_f
    end_position = end_frame / n_f
    covers_full = (1.0 - start_position) * end_position
    
    # Neuroacoustic
    c_pre = chroma_pre.astype(np.float64)
    c_post = chroma_post.astype(np.float64)
    consonance_pre = float(consonance_model(c_pre))
    consonance_post = float(consonance_model(c_post))
    consonance_diff = abs(consonance_pre - consonance_post)
    roughness_pre = float(roughness_model(c_pre))
    roughness_post = float(roughness_model(c_post))
    harmonic_tension = float(harmonic_tension_between(c_pre, c_post))
    
    # Advanced
    phase_coherence = float(phase_coherence_score(beat_phase_pre, beat_phase_post, sub_phase_pre, sub_phase_post))
    
    # Key match
    key1, mode1, _ = key_detection(c_pre)
    key2, mode2, _ = key_detection(c_post)
    key_match = 1.0 - float(key_distance(key1, mode1, key2, mode2))
    
    # Autocorrelation score (periodicity match)
    autocorr_score = 0.5  # Default
    if ctx > 2:
        onset_pre_ctx = feat.onset_env[max(0, b2 - ctx):b2]
        onset_post_ctx = feat.onset_env[b1:min(b1 + ctx, n_f)]
        if len(onset_pre_ctx) > 2 and len(onset_post_ctx) > 2:
            pre_std = np.std(onset_pre_ctx)
            post_std = np.std(onset_post_ctx)
            if pre_std > 1e-6 and post_std > 1e-6:
                min_len = min(len(onset_pre_ctx), len(onset_post_ctx))
                corr = np.corrcoef(onset_pre_ctx[-min_len:], onset_post_ctx[:min_len])[0, 1]
                if not np.isnan(corr):
                    autocorr_score = float((corr + 1) / 2)
    
    return TransitionFeatures(
        chroma_pre=chroma_pre.astype(np.float32),
        chroma_post=chroma_post.astype(np.float32),
        mfcc_pre=mfcc_pre.astype(np.float32),
        mfcc_post=mfcc_post.astype(np.float32),
        tonnetz_pre=tonnetz_pre.astype(np.float32),
        tonnetz_post=tonnetz_post.astype(np.float32),
        rms_pre=float(rms_pre),
        rms_post=float(rms_post),
        rms_diff=float(rms_diff),
        rms_ratio=float(rms_ratio),
        centroid_pre=float(centroid_pre),
        centroid_post=float(centroid_post),
        centroid_diff=float(centroid_diff),
        flatness_pre=float(flatness_pre),
        flatness_post=float(flatness_post),
        bandwidth_pre=float(bandwidth_pre),
        bandwidth_post=float(bandwidth_post),
        flux_pre=float(flux_pre),
        flux_post=float(flux_post),
        flux_diff=float(flux_diff),
        beat_phase_pre=float(beat_phase_pre),
        beat_phase_post=float(beat_phase_post),
        phase_diff=float(phase_diff),
        sub_phase_pre=float(sub_phase_pre),
        sub_phase_post=float(sub_phase_post),
        sub_phase_diff=float(sub_phase_diff),
        transient_pre=float(transient_pre),
        transient_post=float(transient_post),
        onset_pre=float(onset_pre),
        onset_post=float(onset_post),
        masking_pre=float(masking_pre),
        masking_post=float(masking_post),
        loop_duration=float(loop_duration),
        start_position=float(start_position),
        end_position=float(end_position),
        covers_full=float(covers_full),
        consonance_pre=consonance_pre,
        consonance_post=consonance_post,
        consonance_diff=consonance_diff,
        roughness_pre=roughness_pre,
        roughness_post=roughness_post,
        harmonic_tension=harmonic_tension,
        phase_coherence=phase_coherence,
        autocorr_score=autocorr_score,
        key_match=key_match,
    )


def generate_expert_training_data(feat: Features, n_samples: int = 800):
    """Generate training data with specialized targets for 8 experts."""
    ssm_c = feat.ssm_chroma
    ssm_m = feat.ssm_mfcc
    n = ssm_c.shape[0]
    beats = feat.beats
    n_beats = len(beats)
    
    if n_beats < 12:
        return None
    
    X_list = []
    y_harmonic = []
    y_rhythm = []
    y_timbre = []
    y_dynamics = []
    y_neuroacoustic = []
    y_onset = []
    y_spectral_flow = []
    y_phrase = []
    y_combined = []
    
    n_pairs = min(n_samples, n_beats * (n_beats - 1) // 2)
    ssm_upper_idx = np.triu_indices(n, k=4)
    
    if len(ssm_upper_idx[0]) < n_pairs:
        n_pairs = len(ssm_upper_idx[0])
    
    if n_pairs < 30:
        return None
    
    indices = np.random.choice(len(ssm_upper_idx[0]), size=n_pairs, replace=False)
    
    for idx in indices:
        i, j = ssm_upper_idx[0][idx], ssm_upper_idx[1][idx]
        
        if i >= n_beats or j >= n_beats:
            continue
        
        start_frame = beats[i]
        end_frame = beats[j]
        
        if end_frame - start_frame < feat.beat_length * 2:
            continue
        
        try:
            tf = extract_transition_features(feat, start_frame, end_frame)
            X_list.append(tf.to_vector())
            
            # Harmonic target: SSM chroma + key match + consonance
            harm_score = float(ssm_c[i, j]) * 0.5 + tf.key_match * 0.3 + tf.consonance_post * 0.2
            y_harmonic.append(harm_score)
            
            # Rhythm target: phase coherence + transient avoidance
            phase_score = tf.phase_coherence
            transient_penalty = max(tf.transient_pre, tf.transient_post)
            onset_penalty = max(tf.onset_pre, tf.onset_post)
            rhythm_score = phase_score * (1 - transient_penalty * 0.7) * (1 - onset_penalty * 0.5)
            y_rhythm.append(float(max(0, rhythm_score)))
            
            # Timbre target: MFCC SSM + spectral continuity
            timbre_sim = float(ssm_m[i, j])
            spectral_cont = 1.0 - tf.centroid_diff * 2
            spectral_cont = max(0, min(1, spectral_cont))
            y_timbre.append(timbre_sim * 0.6 + spectral_cont * 0.4)
            
            # Dynamics target: RMS match + ratio
            dyn_score = tf.rms_ratio * 0.6 + (1.0 - tf.rms_diff * 2) * 0.4
            y_dynamics.append(float(max(0, min(1, dyn_score))))
            
            # Neuroacoustic target: consonance match + low roughness + low tension
            cons_match = 1.0 - tf.consonance_diff
            low_rough = 1.0 - (tf.roughness_pre + tf.roughness_post) / 2
            low_tension = 1.0 - tf.harmonic_tension
            y_neuroacoustic.append(float(cons_match * 0.4 + low_rough * 0.3 + low_tension * 0.3))
            
            # Onset target: avoid transients + rhythm flow
            no_transient = 1.0 - max(tf.transient_pre, tf.transient_post)
            no_onset = 1.0 - max(tf.onset_pre, tf.onset_post)
            y_onset.append(float(no_transient * 0.6 + no_onset * 0.4))
            
            # Spectral flow target: low flux + flux match
            low_flux = 1.0 - (tf.flux_pre + tf.flux_post) / 2
            flux_match = 1.0 - tf.flux_diff * 2
            y_spectral_flow.append(float(max(0, low_flux * 0.5 + flux_match * 0.5)))
            
            # Phrase target: coverage + duration
            phrase_score = tf.covers_full * 0.4 + min(1.0, tf.loop_duration * 2) * 0.6
            y_phrase.append(float(phrase_score))
            
            # Combined target: overall quality (for meta-learner training)
            combined = (
                harm_score * 0.15 +
                rhythm_score * 0.25 +
                timbre_sim * 0.10 +
                dyn_score * 0.08 +
                cons_match * 0.12 +
                no_transient * 0.15 +
                low_flux * 0.08 +
                phrase_score * 0.07
            )
            y_combined.append(float(combined))
            
        except Exception:
            continue
    
    if len(X_list) < 30:
        return None
    
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_harmonic, dtype=np.float32),
        np.array(y_rhythm, dtype=np.float32),
        np.array(y_timbre, dtype=np.float32),
        np.array(y_dynamics, dtype=np.float32),
        np.array(y_neuroacoustic, dtype=np.float32),
        np.array(y_onset, dtype=np.float32),
        np.array(y_spectral_flow, dtype=np.float32),
        np.array(y_phrase, dtype=np.float32),
        np.array(y_combined, dtype=np.float32),
    )


class NeuralScorer:
    """Premium neural scorer with 8-expert ensemble + meta-learner."""
    
    def __init__(self, feat: Features):
        self.feat = feat
        self.ensemble = ExpertEnsemble(input_dim=TransitionFeatures.dim())
        self.trained = False
        self.X_mean = None
        self.X_std = None
    
    def train_on_composition(self, n_samples: int = 800, epochs: int = 70):
        """Train the 8-expert ensemble on this composition."""
        logging.info("Training 8-expert ensemble with meta-learner...")
        
        data = generate_expert_training_data(self.feat, n_samples=n_samples)
        
        if data is None:
            logging.warning("Not enough training data")
            return
        
        X, y_h, y_r, y_t, y_d, y_n, y_o, y_sf, y_p, y_c = data
        
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.X_mean) / self.X_std
        
        self.ensemble.train_all(X_norm, y_h, y_r, y_t, y_d, y_n, y_o, y_sf, y_p, y_c, epochs=epochs)
        self.trained = True
        
        logging.info(f"8-expert ensemble trained on {len(X)} examples")
    
    def score_batch(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Score multiple transitions."""
        n = len(starts)
        
        if not self.trained:
            return np.full(n, 0.5, dtype=np.float32)
        
        X = np.zeros((n, TransitionFeatures.dim()), dtype=np.float32)
        
        for i in range(n):
            tf = extract_transition_features(self.feat, int(starts[i]), int(ends[i]))
            X[i] = tf.to_vector()
        
        X_norm = (X - self.X_mean) / self.X_std
        
        return self.ensemble.forward_batch(X_norm, use_meta=True).astype(np.float32)


# ============================================================================
# RULE-BASED SCORING (supplements neural)
# ============================================================================

@njit(cache=True, fastmath=True)
def score_harmonic_progression(chroma: np.ndarray, tonnetz: np.ndarray, start: int, end: int) -> float:
    """Score harmonic/tonal transition quality."""
    n_f = chroma.shape[1]
    b1 = min(max(0, start), n_f - 1)
    b2 = min(max(0, end - 1), n_f - 1)
    
    c1 = chroma[:, b1]
    c2 = chroma[:, b2]
    
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for i in range(12):
        dot += c1[i] * c2[i]
        n1 += c1[i] * c1[i]
        n2 += c2[i] * c2[i]
    
    denom = np.sqrt(n1 * n2)
    chroma_sim = dot / denom if denom > 1e-10 else 0.5
    
    t1 = tonnetz[:, b1]
    t2 = tonnetz[:, b2]
    
    tonnetz_dist = 0.0
    for i in range(6):
        tonnetz_dist += (t1[i] - t2[i]) ** 2
    tonnetz_sim = 1.0 - np.sqrt(tonnetz_dist) / 2
    tonnetz_sim = max(0, tonnetz_sim)
    
    return min(1.0, chroma_sim * 0.5 + tonnetz_sim * 0.5)


def score_neuroacoustic_batch(feat: Features, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Score using neuroacoustic principles."""
    n = len(starts)
    scores = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        b1 = min(max(0, int(starts[i])), feat.n_frames - 1)
        b2 = min(max(0, int(ends[i]) - 1), feat.n_frames - 1)
        
        c_pre = feat.chroma_cens[:, b2].astype(np.float64)
        c_post = feat.chroma_cens[:, b1].astype(np.float64)
        
        cons_pre = consonance_model(c_pre)
        cons_post = consonance_model(c_post)
        cons_match = 1.0 - abs(cons_pre - cons_post)
        
        rough_pre = roughness_model(c_pre)
        rough_post = roughness_model(c_post)
        low_rough = 1.0 - (rough_pre + rough_post) / 2
        
        tension = harmonic_tension_between(c_pre, c_post)
        low_tension = 1.0 - tension
        
        key1, mode1, _ = key_detection(c_pre)
        key2, mode2, _ = key_detection(c_post)
        key_match = 1.0 - key_distance(key1, mode1, key2, mode2)
        
        scores[i] = cons_match * 0.25 + low_rough * 0.20 + low_tension * 0.25 + key_match * 0.30
    
    return scores


def score_composition_coverage(
    starts: np.ndarray,
    ends: np.ndarray,
    n_frames: int,
    beats: np.ndarray,
    bar_length: int,
) -> np.ndarray:
    """Score how well the loop covers the composition."""
    n = len(starts)
    scores = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        start_pos = starts[i] / n_frames
        end_pos = ends[i] / n_frames
        duration = (ends[i] - starts[i]) / n_frames
        
        early_start = max(0, 1.0 - start_pos * 5) if start_pos < 0.2 else 0
        late_end = max(0, (end_pos - 0.8) * 5) if end_pos > 0.8 else 0
        coverage_bonus = min(1.0, duration * 1.5) if duration > 0.5 else duration
        
        short_penalty = 0
        if duration < 0.15:
            short_penalty = (0.15 - duration) * 2
        
        loop_len = ends[i] - starts[i]
        bars = loop_len / bar_length if bar_length > 0 else 0
        bar_aligned = 0
        for target in [4, 8, 12, 16, 24, 32]:
            if abs(bars - target) < 0.5:
                bar_aligned = 0.15
                break
        
        scores[i] = (
            early_start * 0.10 +
            late_end * 0.10 +
            coverage_bonus * 0.25 +
            bar_aligned * 0.15 +
            0.40
        ) - short_penalty
        
        scores[i] = max(0, min(1, scores[i]))
    
    return scores


def compute_premium_scores(
    feat: Features,
    starts: np.ndarray,
    ends: np.ndarray,
    neural_scorer: NeuralScorer | None = None,
) -> np.ndarray:
    """Compute premium scores - HARMONIC focus (rhythm in base)."""
    n = len(starts)
    
    harmonic = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s, e = int(starts[i]), int(ends[i])
        harmonic[i] = score_harmonic_progression(feat.chroma_cens, feat.tonnetz, s, e)
    
    neuro_rules = score_neuroacoustic_batch(feat, starts, ends)
    coverage = score_composition_coverage(starts, ends, feat.n_frames, feat.beats, feat.bar_length)
    
    if neural_scorer is not None and neural_scorer.trained:
        neural = neural_scorer.score_batch(starts, ends)
    else:
        neural = np.full(n, 0.5, dtype=np.float32)
    
    combined = (
        harmonic * 0.22 +
        neuro_rules * 0.20 +
        coverage * 0.18 +
        neural * 0.40
    )
    
    return combined


def apply_neural_enhancement(
    feat: Features,
    candidates: list,
    base_scores: np.ndarray,
    use_neural: bool = True,
) -> np.ndarray:
    """Apply premium neural enhancement (rhythm-preserving)."""
    starts = np.array([c._loop_start_frame_idx for c in candidates], dtype=np.int64)
    ends = np.array([c._loop_end_frame_idx for c in candidates], dtype=np.int64)
    
    neural_scorer = None
    if use_neural:
        neural_scorer = NeuralScorer(feat)
        neural_scorer.train_on_composition(n_samples=800, epochs=70)
    
    premium = compute_premium_scores(feat, starts, ends, neural_scorer)
    
    # Neural only ENHANCES good base scores
    enhanced = np.copy(base_scores)
    for i in range(len(base_scores)):
        if base_scores[i] > 0.35:  # Only enhance decent candidates
            enhanced[i] = base_scores[i] * 0.80 + premium[i] * 0.20
    
    return enhanced
