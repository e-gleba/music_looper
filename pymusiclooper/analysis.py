"""
PyMusicLooper Analysis Module - Facade for backwards compatibility.

This module re-exports the modular analysis components from the analysis/ package.
All functionality has been refactored into specialized modules:

analysis/
├── features.py          - Audio feature extraction
├── candidates.py         - Loop candidate generation
├── scoring.py            - Expert-based scoring
├── segment_loop.py       - User segment looping
├── main.py               - Main orchestration
└── experts/              - 8-expert ensemble
    ├── harmonic.py       - Tonal/harmonic transitions
    ├── rhythm.py         - Beat/timing precision
    ├── timbre.py         - Spectral continuity
    ├── psychoacoustic.py - Perceptual masking & JND
    ├── neuroacoustic.py  - Consonance & roughness
    ├── dynamics.py       - Energy flow
    ├── onset.py          - Transient handling
    ├── phrase.py         - Musical structure
    └── ensemble.py       - Meta-learner ensemble

For new code, prefer importing from the analysis package directly:
    from pymusiclooper.analysis import find_best_loop_points, Features

For segment-based looping (user selects a favorite moment):
    from pymusiclooper.analysis import find_segment_loop_points
"""

# Re-export everything for backwards compatibility
from pymusiclooper.analysis.features import (
    Features,
    extract_features,
    nearest_zero_crossing,
    HOP_LENGTH,
    N_FFT,
)

from pymusiclooper.analysis.candidates import (
    LoopPair,
    find_candidates,
    prune_candidates,
)

from pymusiclooper.analysis.scoring import (
    score_candidates,
    finalize_with_alignment,
    optimize_selection,
)

from pymusiclooper.analysis.main import (
    find_best_loop_points,
    find_loop_for_segment,
)

from pymusiclooper.analysis.segment_loop import (
    find_segment_loop_points,
    SegmentLoopResult,
    find_repetitions_in_segment,
)

# Expert ensemble (for advanced usage)
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
    ExpertEnsemble,
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
    'ExpertEnsemble',
    
    # Constants
    'HOP_LENGTH',
    'N_FFT',
    
    # Utilities
    'nearest_zero_crossing',
    'prune_candidates',
    'optimize_selection',
]
