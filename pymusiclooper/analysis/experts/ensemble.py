"""
Expert Ensemble with Meta-Learning.

Combines all domain experts with adaptive weighting:
- Static weights based on expert reliability
- Meta-learner that learns optimal weights per composition
- Gating mechanism for rhythm-critical decisions
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext
from pymusiclooper.analysis.experts.harmonic import HarmonicExpert
from pymusiclooper.analysis.experts.rhythm import RhythmExpert
from pymusiclooper.analysis.experts.timbre import TimbreExpert
from pymusiclooper.analysis.experts.psychoacoustic import PsychoacousticExpert
from pymusiclooper.analysis.experts.neuroacoustic import NeuroacousticExpert
from pymusiclooper.analysis.experts.dynamics import DynamicsExpert
from pymusiclooper.analysis.experts.onset import OnsetExpert
from pymusiclooper.analysis.experts.phrase import PhraseExpert
from pymusiclooper.analysis.experts.melodic import MelodicExpert
from pymusiclooper.analysis.experts.spectral import SpectralExpert
from pymusiclooper.analysis.experts.crossfade import CrossfadeExpert
from pymusiclooper.analysis.experts.microtiming import MicrotimingExpert
from pymusiclooper.analysis.experts.energy_flow import EnergyFlowExpert
from pymusiclooper.analysis.experts.resonance import ResonanceExpert
from pymusiclooper.analysis.experts.texture import TextureExpert
from pymusiclooper.analysis.experts.continuity import ContinuityExpert
from pymusiclooper.analysis.experts.vocal import VocalExpert
from pymusiclooper.analysis.experts.ensemble_instruments import EnsembleInstrumentsExpert
from pymusiclooper.analysis.experts.transition_effects import TransitionEffectsExpert

if TYPE_CHECKING:
    from pymusiclooper.analysis.features import Features


class ExpertEnsemble:
    """
    16-Expert Ensemble with adaptive weighting and rhythm gating.
    
    The ensemble combines specialized experts with a meta-learner
    that adjusts weights per composition. A gating mechanism ensures
    rhythm alignment is never compromised.
    
    Core Experts:
    0. HarmonicExpert    - Tonal/harmonic transitions
    1. RhythmExpert      - Beat/timing precision (HIGH weight)
    2. TimbreExpert      - Spectral continuity
    3. PsychoacousticExpert - Perceptual masking
    4. NeuroacousticExpert - Consonance/roughness
    5. DynamicsExpert    - Energy flow
    6. OnsetExpert       - Transient handling (HIGH weight)
    7. PhraseExpert      - Musical structure
    8. MelodicExpert  - Melodic contour
    9. SpectralExpert    - Deep spectral analysis
    10. CrossfadeExpert  - Waveform quality
    
    Organic Flow Experts (NEW):
    11. MicrotimingExpert - Groove, swing, micro-timing
    12. EnergyFlowExpert  - Energy envelope continuity
    13. ResonanceExpert  - Formant and resonance analysis
    14. TextureExpert     - Spectral texture and density
    15. ContinuityExpert  - Overall seamlessness (HIGH weight)
    16. VocalExpert       - Vocal pause and word boundary detection (HIGH weight)
    17. EnsembleInstrumentsExpert - Ensemble instrument continuity (HIGH weight)
    18. TransitionEffectsExpert - Transition effects optimization (HIGH weight)
    """
    
    def __init__(self):
        # Initialize all 19 experts
        self.experts: list[Expert] = [
            HarmonicExpert(),      # 0
            RhythmExpert(),        # 1
            TimbreExpert(),        # 2
            PsychoacousticExpert(),# 3
            NeuroacousticExpert(), # 4
            DynamicsExpert(),      # 5
            OnsetExpert(),         # 6
            PhraseExpert(),        # 7
            MelodicExpert(),       # 8
            SpectralExpert(),      # 9
            CrossfadeExpert(),     # 10
            MicrotimingExpert(),   # 11
            EnergyFlowExpert(),    # 12
            ResonanceExpert(),    # 13
            TextureExpert(),      # 14
            ContinuityExpert(),    # 15
            VocalExpert(),        # 16
            EnsembleInstrumentsExpert(),  # 17
            TransitionEffectsExpert(),  # 18
        ]
        
        # Base weights (can be overridden by meta-learner)
        self.base_weights = np.array([
            expert.weight for expert in self.experts
        ], dtype=np.float32)
        
        # Normalize to sum to 1
        self.base_weights /= np.sum(self.base_weights)
        
        # Adaptive weights (learned per composition)
        self.adaptive_weights = None
        
        # Gating thresholds
        self.rhythm_gate_threshold = 0.4
        self.onset_gate_threshold = 0.3
    
    def score(self, ctx: TransitionContext) -> float:
        """
        Score a single transition using all experts.
        
        Returns a score in [0, 1] combining all expert opinions
        with rhythm/onset gating for safety.
        """
        # Get individual expert scores
        expert_scores = np.array([
            expert.score(ctx) for expert in self.experts
        ], dtype=np.float32)
        
        # Use adaptive weights if trained, else base weights
        weights = self.adaptive_weights if self.adaptive_weights is not None else self.base_weights
        
        # Weighted combination
        combined = np.sum(expert_scores * weights)
        
        # Apply gating for critical experts
        rhythm_score = expert_scores[1]   # RhythmExpert is index 1
        onset_score = expert_scores[6]    # OnsetExpert is index 6
        melodic_score = expert_scores[8] if len(expert_scores) > 8 else 1.0  # MelodicExpert
        continuity_score = expert_scores[15] if len(expert_scores) > 15 else 1.0  # ContinuityExpert
        
        gate = self._compute_gate(rhythm_score, onset_score, melodic_score, continuity_score)
        
        return float(combined * gate)
    
    def score_batch(
        self,
        feat: Features,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        """Score multiple transitions efficiently."""
        n = len(starts)
        scores = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            ctx = TransitionContext.from_features(feat, int(starts[i]), int(ends[i]))
            scores[i] = self.score(ctx)
        
        return scores
    
    def _compute_gate(
        self, 
        rhythm_score: float, 
        onset_score: float, 
        melodic_score: float = 1.0,
        continuity_score: float = 1.0
    ) -> float:
        """
        Compute gating multiplier based on critical scores.
        
        Rhythm and onset alignment are critical - bad values in these
        areas should significantly reduce the overall score.
        Melodic continuity and overall continuity are also considered.
        """
        gate = 1.0
        
        # Rhythm gate (most critical)
        if rhythm_score < self.rhythm_gate_threshold:
            gate *= 0.15 + rhythm_score * 0.5
        elif rhythm_score < 0.6:
            gate *= 0.5 + rhythm_score * 0.5
        
        # Onset gate (critical)
        if onset_score < self.onset_gate_threshold:
            gate *= 0.2 + onset_score * 0.5
        
        # Melodic penalty (less severe)
        if melodic_score < 0.3:
            gate *= 0.7 + melodic_score
        
        # Continuity gate (important for seamlessness)
        if continuity_score < 0.4:
            gate *= 0.3 + continuity_score * 0.7
        elif continuity_score < 0.6:
            gate *= 0.6 + continuity_score * 0.4
        
        return gate
    
    def train_on_composition(
        self,
        feat: Features,
        n_samples: int = 500,
        epochs: int = 50,
    ):
        """
        Train adaptive weights for this specific composition.
        
        Uses self-supervision: generate transition examples and
        learn weights that produce a smooth scoring distribution.
        """
        logging.info("Training expert ensemble for composition...")
        
        beats = feat.beats
        n_beats = len(beats)
        
        if n_beats < 12:
            logging.warning("Not enough beats for training")
            return
        
        # Generate training samples
        samples = []
        targets = []
        
        # Sample beat pairs
        n_pairs = min(n_samples, n_beats * (n_beats - 1) // 2)
        
        for _ in range(n_pairs):
            i = np.random.randint(0, n_beats - 4)
            j = np.random.randint(i + 4, min(i + 50, n_beats))
            
            start_frame = int(beats[i])
            end_frame = int(beats[j])
            
            if end_frame - start_frame < feat.beat_length * 2:
                continue
            
            try:
                ctx = TransitionContext.from_features(feat, start_frame, end_frame)
                
                # Get expert scores - vectorized
                expert_scores = np.array([expert.score(ctx) for expert in self.experts], dtype=np.float32)
                
                samples.append(expert_scores)
                
                # Create target from SSM similarity
                ssm_score = feat.ssm_chroma[i % len(feat.ssm_chroma), j % len(feat.ssm_chroma[0])]
                targets.append(float(ssm_score))
                
            except Exception:
                continue
        
        if len(samples) < 30:
            logging.warning("Not enough training samples")
            return
        
        X = np.array(samples, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        # Improved gradient descent with adaptive learning rate
        weights = self.base_weights.copy()
        lr = 0.02  # Slightly higher initial learning rate
        best_loss = float('inf')
        patience = 10
        no_improve = 0
        
        for epoch in range(epochs):
            # Predictions
            pred = X @ weights
            
            # MSE loss
            error = pred - y
            loss = np.mean(error ** 2)
            
            # Early stopping if no improvement
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logging.debug(f"Early stopping at epoch {epoch}")
                    break
            
            # Adaptive learning rate
            if epoch > 0 and epoch % 20 == 0:
                lr *= 0.9  # Decay learning rate
            
            # Gradient
            grad = 2 * X.T @ error / len(y)
            
            # Update (with L2 regularization toward base weights)
            weights -= lr * grad
            weights -= 0.01 * (weights - self.base_weights)
            
            # Ensure non-negative and normalized
            weights = np.maximum(weights, 0.01)
            weights /= np.sum(weights)
        
        self.adaptive_weights = weights
        logging.info(f"Trained adaptive weights: {weights}")
    
    def explain_score(self, ctx: TransitionContext) -> str:
        """Get detailed explanation of score breakdown."""
        lines = [f"Ensemble Score: {self.score(ctx):.3f}"]
        lines.append("-" * 40)
        
        for expert in self.experts:
            score = expert.score(ctx)
            weight = self.base_weights[self.experts.index(expert)]
            lines.append(f"  {expert.name:15} | Score: {score:.2f} | Weight: {weight:.2f}")
        
        return "\n".join(lines)


class WaveformCrossfadeScorer:
    """
    Sample-level crossfade quality scorer.
    
    Analyzes waveform compatibility at transition points
    for the final quality assessment.
    """
    
    def __init__(self, audio: np.ndarray, sr: int, hop: int):
        self.audio = audio
        self.sr = sr
        self.hop = hop
    
    def score(
        self,
        start_frame: int,
        end_frame: int,
        fade_ms: int = 50,
    ) -> float:
        """Score waveform compatibility for crossfade."""
        fade_samples = int(self.sr * fade_ms / 1000)
        
        # Convert frames to samples
        start_samples = start_frame * self.hop
        end_samples = end_frame * self.hop
        
        # Get audio regions
        pre_end = end_samples
        pre_start = max(0, pre_end - fade_samples)
        post_start = start_samples
        post_end = min(len(self.audio), post_start + fade_samples)
        
        fade_len = min(pre_end - pre_start, post_end - post_start)
        
        if fade_len < 32:
            return 0.3
        
        pre = self.audio[pre_end - fade_len:pre_end]
        post = self.audio[post_start:post_start + fade_len]
        
        # 1. Waveform correlation
        std_pre = np.std(pre)
        std_post = np.std(post)
        
        if std_pre > 1e-8 and std_post > 1e-8:
            corr = np.corrcoef(pre, post)[0, 1]
            corr = 0.0 if np.isnan(corr) else max(0, corr)
        else:
            corr = 0.3
        
        # 2. Energy envelope match
        n_windows = min(8, fade_len // 64)
        if n_windows >= 2:
            win_size = fade_len // n_windows
            pre_env = np.array([
                np.mean(pre[j*win_size:(j+1)*win_size]**2) 
                for j in range(n_windows)
            ])
            post_env = np.array([
                np.mean(post[j*win_size:(j+1)*win_size]**2) 
                for j in range(n_windows)
            ])
            
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
        
        # 3. Zero-crossing rate match
        zcr_pre = np.sum(np.abs(np.diff(np.sign(pre)))) / (2 * fade_len + 1e-10)
        zcr_post = np.sum(np.abs(np.diff(np.sign(post)))) / (2 * fade_len + 1e-10)
        zcr_match = 1.0 - min(abs(zcr_pre - zcr_post) * 3, 1.0)
        
        # 4. Low amplitude at cut point
        cut_amp_pre = abs(pre[-1]) / (np.max(np.abs(pre)) + 1e-10)
        cut_amp_post = abs(post[0]) / (np.max(np.abs(post)) + 1e-10)
        low_amp = 1.0 - (cut_amp_pre + cut_amp_post) / 2
        
        # 5. Smoothness of crossfade
        fade_out = np.linspace(1, 0, min(fade_len, 256))
        fade_in = np.linspace(0, 1, min(fade_len, 256))
        mixed = pre[-len(fade_out):] * fade_out + post[:len(fade_in)] * fade_in
        smoothness = 1.0 / (1.0 + np.std(np.diff(mixed)) * 20)
        
        return (
            corr * 0.25 + 
            env_corr * 0.25 + 
            zcr_match * 0.20 + 
            low_amp * 0.15 + 
            smoothness * 0.15
        )

