"""
Neural Scoring Module - Facade for backwards compatibility.

This module has been refactored into the analysis/experts/ package.
All exports are re-exported here for backwards compatibility.

For new code, use:
    from pymusiclooper.analysis.experts import ExpertEnsemble
    from pymusiclooper.analysis.experts.neuroacoustic import consonance_model, roughness_model
"""

# Re-export from new location for backwards compatibility
from pymusiclooper.analysis.experts.base import (
    Expert,
    TransitionContext,
    cosine_similarity,
    circular_distance,
)

from pymusiclooper.analysis.experts.harmonic import (
    HarmonicExpert,
    detect_key,
    compute_key_distance,
    compute_harmonic_tension,
)

from pymusiclooper.analysis.experts.rhythm import (
    RhythmExpert,
    compute_phase_coherence,
    score_duration_alignment,
)

from pymusiclooper.analysis.experts.neuroacoustic import (
    NeuroacousticExpert,
    consonance_model,
    roughness_model,
    harmonic_tension_model,
    estimate_harmonicity,
    compute_gestalt_continuity,
    score_tension_resolution,
    TensionProfiler,
)

from pymusiclooper.analysis.experts.psychoacoustic import (
    PsychoacousticExpert,
    compute_temporal_masking_score,
    hz_to_bark,
    hz_to_erb,
    compute_spreading_function,
    compute_masking_threshold,
    compute_absolute_threshold,
    CriticalBandAnalyzer,
    BARK_EDGES,
    ERB_CONST,
    ERB_FACTOR,
    JND_LOUDNESS_DB,
    JND_PITCH_CENTS,
    JND_TIMING_MS,
    FORWARD_MASKING_MS,
    BACKWARD_MASKING_MS,
)

from pymusiclooper.analysis.experts.timbre import TimbreExpert
from pymusiclooper.analysis.experts.dynamics import DynamicsExpert
from pymusiclooper.analysis.experts.onset import OnsetExpert
from pymusiclooper.analysis.experts.phrase import PhraseExpert

from pymusiclooper.analysis.experts.ensemble import (
    ExpertEnsemble,
    WaveformCrossfadeScorer,
)


def apply_neural_enhancement(feat, candidates, base_scores, use_neural=True):
    """
    Apply neural enhancement to base scores.
    
    This is a compatibility shim - the new system uses the ExpertEnsemble
    directly in the scoring pipeline.
    """
    import numpy as np
    from pymusiclooper.analysis.experts.ensemble import ExpertEnsemble
    from pymusiclooper.analysis.experts.base import TransitionContext
    
    if not use_neural:
        return base_scores
    
    ensemble = ExpertEnsemble()
    
    # Train on composition
    try:
        ensemble.train_on_composition(feat, n_samples=500, epochs=50)
    except Exception:
        pass
    
    # Score candidates
    starts = np.array([c._loop_start_frame_idx for c in candidates], dtype=np.int64)
    ends = np.array([c._loop_end_frame_idx for c in candidates], dtype=np.int64)
    
    enhanced = np.copy(base_scores)
    
    for i in range(len(candidates)):
        ctx = TransitionContext.from_features(feat, int(starts[i]), int(ends[i]))
        neural_score = ensemble.score(ctx)
        
        # Blend with base score
        if base_scores[i] > 0.35:
            enhanced[i] = base_scores[i] * 0.75 + neural_score * 0.25
    
    return enhanced


# Legacy class aliases for compatibility
NeuralScorer = ExpertEnsemble


__all__ = [
    # Base
    'Expert',
    'TransitionContext',
    'cosine_similarity',
    'circular_distance',
    
    # Experts
    'HarmonicExpert',
    'RhythmExpert',
    'TimbreExpert',
    'PsychoacousticExpert',
    'NeuroacousticExpert',
    'DynamicsExpert',
    'OnsetExpert',
    'PhraseExpert',
    'ExpertEnsemble',
    
    # Psychoacoustic
    'consonance_model',
    'roughness_model',
    'harmonic_tension_model',
    'detect_key',
    'compute_key_distance',
    'compute_harmonic_tension',
    'compute_phase_coherence',
    'hz_to_bark',
    'hz_to_erb',
    'CriticalBandAnalyzer',
    
    # Legacy
    'apply_neural_enhancement',
    'NeuralScorer',
]
