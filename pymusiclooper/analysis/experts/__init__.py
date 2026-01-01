"""
Expert Ensemble for Loop Quality Assessment.

11-Expert ensemble covering all aspects of audio loop quality:

Core Experts:
- HarmonicExpert: Tonal/key transitions, circle of fifths
- RhythmExpert: Beat phase, timing precision
- TimbreExpert: Spectral envelope, MFCC continuity
- DynamicsExpert: Energy flow, loudness matching
- OnsetExpert: Transient handling, attack continuity
- PhraseExpert: Musical structure, phrase boundaries

Perceptual Experts:
- PsychoacousticExpert: Masking, critical bands, JND
- NeuroacousticExpert: Consonance, roughness, tension

Advanced Experts:
- MelodicExpert: Melodic contour, pitch salience
- SpectralExpert: Deep spectral analysis, entropy
- CrossfadeExpert: Waveform-level transition quality
"""

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, cosine_similarity, circular_distance
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
from pymusiclooper.analysis.experts.ensemble import ExpertEnsemble

__all__ = [
    # Base
    'Expert',
    'TransitionContext',
    'cosine_similarity',
    'circular_distance',
    
    # Core experts
    'HarmonicExpert',
    'RhythmExpert',
    'TimbreExpert',
    'DynamicsExpert',
    'OnsetExpert',
    'PhraseExpert',
    
    # Perceptual experts
    'PsychoacousticExpert',
    'NeuroacousticExpert',
    
    # Advanced experts
    'MelodicExpert',
    'SpectralExpert',
    'CrossfadeExpert',
    
    # Ensemble
    'ExpertEnsemble',
]
