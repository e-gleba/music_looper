"""
PyMusicLooper Analysis Module - Premium Audio Loop Detection.

Architecture:
├── features.py          - Audio feature extraction (STFT, chroma, MFCC, etc.)
├── candidates.py        - Loop candidate generation (SSM, beat-aligned)
├── scoring.py           - Expert-based scoring
├── segment_loop.py      - User segment looping algorithm
├── main.py              - Main orchestration
└── experts/             - 8-Expert ensemble for scoring
    ├── base.py          - Abstract expert interface
    ├── harmonic.py      - Harmonic/tonal expert
    ├── rhythm.py        - Rhythm/timing expert
    ├── timbre.py        - Timbre/spectral expert
    ├── psychoacoustic.py - Psychoacoustic perception expert
    ├── neuroacoustic.py  - Neuroacoustic consonance/roughness
    ├── dynamics.py      - Dynamics/energy expert
    ├── onset.py         - Onset/transient expert
    ├── phrase.py        - Musical phrase structure expert
    └── ensemble.py      - Expert ensemble + meta-learner
"""

# Core features and extraction
from pymusiclooper.analysis.features import (
    Features, 
    extract_features, 
    nearest_zero_crossing,
    HOP_LENGTH, 
    N_FFT,
)

# Candidate generation
from pymusiclooper.analysis.candidates import (
    LoopPair, 
    find_candidates, 
    prune_candidates,
)

# Scoring
from pymusiclooper.analysis.scoring import (
    score_candidates, 
    finalize_with_alignment, 
    optimize_selection,
)

# Main entry points
from pymusiclooper.analysis.main import (
    find_best_loop_points,
    find_loop_for_segment,
)

# Segment looping
from pymusiclooper.analysis.segment_loop import (
    find_segment_loop_points,
    SegmentLoopResult,
    find_repetitions_in_segment,
)

# Expert ensemble
from pymusiclooper.analysis.experts import (
    Expert,
    TransitionContext,
    HarmonicExpert,
    RhythmExpert,
    TimbreExpert,
    PsychoacousticExpert,
    NeuroacousticExpert,
    DynamicsExpert,
    OnsetExpert,
    PhraseExpert,
    MelodicExpert,
    SpectralExpert,
    CrossfadeExpert,
    MicrotimingExpert,
    EnergyFlowExpert,
    ResonanceExpert,
    TextureExpert,
    ContinuityExpert,
    ExpertEnsemble,
)

# Quality metrics
from pymusiclooper.analysis.quality import (
    LoopQualityMetrics,
    compute_quality_metrics,
    compute_segment_melodicity,
    rank_loops_by_quality,
)


__all__ = [
    # Core
    'Features',
    'LoopPair',
    'extract_features',
    'find_best_loop_points',
    'find_candidates',
    'score_candidates',
    'finalize_with_alignment',
    
    # Segment looping
    'find_segment_loop_points',
    'find_loop_for_segment',
    'SegmentLoopResult',
    'find_repetitions_in_segment',
    
    # Experts
    'Expert',
    'TransitionContext',
    'HarmonicExpert',
    'RhythmExpert',
    'TimbreExpert',
    'PsychoacousticExpert',
    'NeuroacousticExpert',
    'DynamicsExpert',
    'OnsetExpert',
    'PhraseExpert',
    'MelodicExpert',
    'SpectralExpert',
    'CrossfadeExpert',
    'MicrotimingExpert',
    'EnergyFlowExpert',
    'ResonanceExpert',
    'TextureExpert',
    'ContinuityExpert',
    'ExpertEnsemble',
    
    # Quality metrics
    'LoopQualityMetrics',
    'compute_quality_metrics',
    'compute_segment_melodicity',
    'rank_loops_by_quality',
    
    # Constants
    'HOP_LENGTH',
    'N_FFT',
    
    # Utilities
    'nearest_zero_crossing',
    'prune_candidates',
    'optimize_selection',
]
