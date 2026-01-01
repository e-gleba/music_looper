"""
Analysis Constants - All thresholds and parameters.

Centralized configuration for all analysis parameters to avoid magic numbers
and enable easy tuning.
"""

from __future__ import annotations

# ============================================================================
# SIMILARITY THRESHOLDS
# ============================================================================

# Chroma (harmonic) similarity thresholds
CHROMA_SIM_MIN = 0.4  # Minimum chroma similarity for candidate generation
CHROMA_SIM_GOOD = 0.6  # Good chroma match
CHROMA_SIM_EXCELLENT = 0.8  # Excellent chroma match

# MFCC (timbre) similarity thresholds
MFCC_SIM_MIN = 0.35  # Minimum timbre similarity (reject if below)
MFCC_SIM_ACCEPTABLE = 0.38  # Acceptable timbre match
MFCC_SIM_GOOD = 0.5  # Good timbre match
MFCC_SIM_EXCELLENT = 0.7  # Excellent timbre match

# Combined similarity thresholds
COMBINED_SIM_MIN = 0.45  # Minimum combined similarity for candidates
COMBINED_SIM_GOOD = 0.6  # Good combined match

# Onset pattern similarity
ONSET_SIM_MIN = 0.3  # Minimum onset pattern similarity

# ============================================================================
# TIMBRE FILTERING (for preventing instrument mismatches)
# ============================================================================

TIMBRE_PENALTY_VERY_BAD = 0.25  # Very bad timbre match threshold
TIMBRE_PENALTY_BAD = 0.4  # Bad timbre match threshold
TIMBRE_PENALTY_MULT_VERY_BAD = 0.7  # Score multiplier for very bad timbre
TIMBRE_PENALTY_MULT_BAD = 0.85  # Score multiplier for bad timbre

# ============================================================================
# RHYTHM ALIGNMENT
# ============================================================================

# Beat phase alignment thresholds (0-1, where 0/1 = on beat, 0.5 = off beat)
BEAT_PHASE_EXCELLENT = 0.04  # Excellent alignment (< 4% off beat)
BEAT_PHASE_GOOD = 0.08  # Good alignment (< 8% off beat)
BEAT_PHASE_ACCEPTABLE = 0.15  # Acceptable alignment (< 15% off beat)
BEAT_PHASE_ON_BEAT_THRESHOLD = 0.06  # Considered "on beat" if < 6% or > 94%

# Sub-beat (16th note) alignment
SUB_BEAT_EXCELLENT = 0.06  # Excellent sub-beat alignment
SUB_BEAT_GOOD = 0.12  # Good sub-beat alignment

# ============================================================================
# TRANSIENT/ONSET HANDLING (for drums/percussion)
# ============================================================================

# Transient strength thresholds (0-1)
TRANSIENT_VERY_HIGH = 0.7  # Very high transient (drum hit) - avoid cutting here
TRANSIENT_HIGH = 0.5  # High transient
TRANSIENT_MEDIUM = 0.3  # Medium transient

# Onset peak thresholds
ONSET_PEAK_HIGH = 0.5  # High onset peak - avoid cutting here

# ============================================================================
# ZERO-CROSSING ALIGNMENT
# ============================================================================

# Search window for zero-crossing alignment (milliseconds)
ZERO_CROSSING_SEARCH_MS_BASE = 3.0  # Base search window
ZERO_CROSSING_SEARCH_MS_ON_BEAT = 2.0  # Tighter window when on beat
ZERO_CROSSING_MAX_SHIFT_RATIO = 0.5  # Max acceptable shift (50% of search window)

# ============================================================================
# SCORING WEIGHTS
# ============================================================================

# Ensemble vs crossfade scoring
ENSEMBLE_WEIGHT = 0.75  # Weight for expert ensemble score
CROSSFADE_WEIGHT = 0.25  # Weight for waveform crossfade score

# Combined similarity weights (for candidate generation)
SIM_WEIGHT_CHROMA = 0.4  # Chroma weight in combined similarity
SIM_WEIGHT_MFCC = 0.35  # MFCC weight in combined similarity
SIM_WEIGHT_ONSET = 0.25  # Onset pattern weight in combined similarity

# ============================================================================
# CONTEXT WINDOWS
# ============================================================================

# Context window sizes (in frames) for feature averaging
MFCC_CONTEXT_WINDOW = 10  # Frames for MFCC context
ONSET_CONTEXT_WINDOW = 20  # Frames for onset pattern context

# ============================================================================
# ADAPTIVE THRESHOLDS (computed from audio characteristics)
# ============================================================================

def compute_adaptive_timbre_threshold(avg_transient_strength: float) -> float:
    """
    Compute adaptive timbre threshold based on transient strength.
    
    For music with heavy drums (high transients), be more lenient
    with timbre matching since drums can mask timbre differences.
    """
    if avg_transient_strength > 0.6:
        # Heavy drums - more lenient
        return MFCC_SIM_MIN - 0.05  # 0.30
    elif avg_transient_strength > 0.4:
        # Moderate drums
        return MFCC_SIM_MIN - 0.02  # 0.33
    else:
        # Light drums - standard threshold
        return MFCC_SIM_MIN


def compute_adaptive_rhythm_tolerance(bpm: float) -> float:
    """
    Compute adaptive rhythm tolerance based on tempo.
    
    Faster tempos allow slightly more phase tolerance.
    """
    if bpm > 140:
        # Fast tempo - slightly more tolerance
        return BEAT_PHASE_ACCEPTABLE * 1.2  # ~0.18
    elif bpm < 80:
        # Slow tempo - stricter
        return BEAT_PHASE_ACCEPTABLE * 0.9  # ~0.135
    else:
        # Normal tempo
        return BEAT_PHASE_ACCEPTABLE


def compute_adaptive_zero_crossing_window(
    beat_phase: float,
    transient_strength: float,
    base_window_ms: float = ZERO_CROSSING_SEARCH_MS_BASE
) -> float:
    """
    Compute adaptive zero-crossing search window.
    
    - Tighter window when on beat (preserve rhythm)
    - Tighter window when high transients (preserve drum hits)
    """
    window = base_window_ms
    
    # If on beat, use tighter window
    if beat_phase < BEAT_PHASE_ON_BEAT_THRESHOLD or beat_phase > (1.0 - BEAT_PHASE_ON_BEAT_THRESHOLD):
        window = ZERO_CROSSING_SEARCH_MS_ON_BEAT
    
    # If high transient (drum hit), use tighter window to preserve timing
    if transient_strength > TRANSIENT_MEDIUM:
        window *= 0.8  # 20% tighter
    
    return window

