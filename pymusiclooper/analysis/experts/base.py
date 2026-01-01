"""
Base Expert Interface and Transition Context.

Provides the abstract interface that all domain experts implement,
plus a rich context object containing all relevant transition information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymusiclooper.analysis.features import Features


@dataclass
class TransitionContext:
    """
    Rich context for evaluating a loop transition point.
    
    Contains all information needed by experts to assess transition quality,
    including features at both ends, context windows, and derived metrics.
    """
    
    # Frame indices
    start_frame: int
    end_frame: int
    loop_duration_frames: int
    
    # Chroma at transition points (12-dim)
    chroma_start: np.ndarray
    chroma_end: np.ndarray
    chroma_context_start: np.ndarray  # Mean over context window
    chroma_context_end: np.ndarray
    
    # MFCC at transition points (20-dim)
    mfcc_start: np.ndarray
    mfcc_end: np.ndarray
    mfcc_context_start: np.ndarray
    mfcc_context_end: np.ndarray
    
    # Tonnetz (6-dim)
    tonnetz_start: np.ndarray
    tonnetz_end: np.ndarray
    
    # Spectral
    centroid_start: float
    centroid_end: float
    bandwidth_start: float
    bandwidth_end: float
    flatness_start: float
    flatness_end: float
    flux_start: float
    flux_end: float
    rolloff_start: float
    rolloff_end: float
    
    # Dynamics
    rms_start: float
    rms_end: float
    loudness_start: float
    loudness_end: float
    
    # Rhythm
    beat_phase_start: float
    beat_phase_end: float
    sub_phase_start: float
    sub_phase_end: float
    
    # Transients
    transient_start: float
    transient_end: float
    onset_start: float
    onset_end: float
    
    # Perceptual
    masking_start: float
    masking_end: float
    
    # Context
    n_frames: int
    bar_length: int
    beat_length: int
    bpm: float
    
    # Raw audio references (for waveform analysis)
    audio: np.ndarray
    sr: int
    hop: int
    
    @classmethod
    def from_features(
        cls,
        feat: Features,
        start_frame: int,
        end_frame: int,
        context_frames: int = 15,
    ) -> TransitionContext:
        """Create TransitionContext from Features and frame indices."""
        n_f = feat.n_frames
        ctx = min(context_frames, start_frame, n_f - end_frame)
        ctx = max(ctx, 1)
        
        b1 = min(max(0, start_frame), n_f - 1)
        b2 = min(max(0, end_frame - 1), n_f - 1)
        
        # Get context windows
        start_ctx_range = slice(max(0, b1 - ctx), min(b1 + ctx + 1, n_f))
        end_ctx_range = slice(max(0, b2 - ctx), min(b2 + ctx + 1, n_f))
        
        return cls(
            start_frame=start_frame,
            end_frame=end_frame,
            loop_duration_frames=end_frame - start_frame,
            
            # Chroma
            chroma_start=feat.chroma_cens[:, b1].copy(),
            chroma_end=feat.chroma_cens[:, b2].copy(),
            chroma_context_start=np.mean(feat.chroma_cens[:, start_ctx_range], axis=1),
            chroma_context_end=np.mean(feat.chroma_cens[:, end_ctx_range], axis=1),
            
            # MFCC
            mfcc_start=feat.mfcc[:, b1].copy(),
            mfcc_end=feat.mfcc[:, b2].copy(),
            mfcc_context_start=np.mean(feat.mfcc[:, start_ctx_range], axis=1),
            mfcc_context_end=np.mean(feat.mfcc[:, end_ctx_range], axis=1),
            
            # Tonnetz
            tonnetz_start=feat.tonnetz[:, b1].copy(),
            tonnetz_end=feat.tonnetz[:, b2].copy(),
            
            # Spectral
            centroid_start=float(feat.spectral_centroid[b1]),
            centroid_end=float(feat.spectral_centroid[b2]),
            bandwidth_start=float(feat.spectral_bandwidth[b1]),
            bandwidth_end=float(feat.spectral_bandwidth[b2]),
            flatness_start=float(feat.spectral_flatness[b1]),
            flatness_end=float(feat.spectral_flatness[b2]),
            flux_start=float(feat.spectral_flux[b1]),
            flux_end=float(feat.spectral_flux[b2]),
            rolloff_start=float(feat.spectral_rolloff[b1]),
            rolloff_end=float(feat.spectral_rolloff[b2]),
            
            # Dynamics
            rms_start=float(feat.rms[b1]),
            rms_end=float(feat.rms[b2]),
            loudness_start=float(feat.loudness[b1]),
            loudness_end=float(feat.loudness[b2]),
            
            # Rhythm
            beat_phase_start=float(feat.beat_phase[b1]),
            beat_phase_end=float(feat.beat_phase[b2]),
            sub_phase_start=float(feat.sub_beat_phase[b1]),
            sub_phase_end=float(feat.sub_beat_phase[b2]),
            
            # Transients
            transient_start=float(feat.transient_strength[b1]),
            transient_end=float(feat.transient_strength[b2]),
            onset_start=float(feat.onset_peaks[b1]),
            onset_end=float(feat.onset_peaks[b2]),
            
            # Perceptual
            masking_start=float(feat.masking_curve[b1]),
            masking_end=float(feat.masking_curve[b2]),
            
            # Context
            n_frames=n_f,
            bar_length=feat.bar_length,
            beat_length=feat.beat_length,
            bpm=feat.bpm,
            
            # Audio
            audio=feat.audio,
            sr=feat.sr,
            hop=feat.hop,
        )


class Expert(ABC):
    """
    Abstract base class for domain-specific loop quality experts.
    
    Each expert evaluates loop transitions from a specific perceptual perspective
    and returns a score in [0, 1] where 1 is perfect.
    """
    
    name: str = "base"
    weight: float = 0.1  # Default weight in ensemble
    
    @abstractmethod
    def score(self, ctx: TransitionContext) -> float:
        """
        Evaluate transition quality from this expert's perspective.
        
        Args:
            ctx: Rich context containing all transition information
            
        Returns:
            Score in [0, 1] where 1 is a perfect transition
        """
        pass
    
    def score_batch(
        self, 
        contexts: list[TransitionContext],
    ) -> np.ndarray:
        """Score multiple transitions (can be overridden for vectorized impl)."""
        return np.array([self.score(ctx) for ctx in contexts], dtype=np.float32)
    
    @abstractmethod
    def explain(self, ctx: TransitionContext) -> str:
        """Human-readable explanation of the score."""
        pass


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def circular_distance(phase1: float, phase2: float) -> float:
    """Compute circular distance between two phase values in [0, 1]."""
    diff = abs(phase1 - phase2)
    return min(diff, 1.0 - diff)

