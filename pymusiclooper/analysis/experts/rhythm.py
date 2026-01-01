"""
Rhythm Expert - Beat Timing and Phase Alignment.

Evaluates:
- Beat phase alignment (critical for seamless loops)
- Sub-beat phase alignment (16th note precision)
- Groove continuity
- Tempo consistency
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, circular_distance
from pymusiclooper.analysis.constants import (
    BEAT_PHASE_EXCELLENT,
    BEAT_PHASE_GOOD,
    BEAT_PHASE_ACCEPTABLE,
    BEAT_PHASE_ON_BEAT_THRESHOLD,
    SUB_BEAT_EXCELLENT,
    SUB_BEAT_GOOD,
    compute_adaptive_rhythm_tolerance
)


class RhythmExpert(Expert):
    """Expert for rhythmic alignment and timing precision.
    
    ULTRA-PRECISE rhythm analysis with:
    - Strict beat phase alignment (critical for drums)
    - Tempo consistency checking
    - Beat pattern continuity
    - Groove preservation
    """
    
    name = "rhythm"
    weight = 0.25  # Increased weight - rhythm is CRITICAL for seamless loops
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate rhythmic alignment at transition point.
        
        CRITICAL for drums/rock music - must be precisely aligned to beats.
        Uses ULTRA-STRICT requirements for rhythm-critical music.
        """
        # Check if we have high transients (drums) - need stricter alignment
        avg_transient = (ctx.transient_start + ctx.transient_end) / 2
        has_drums = avg_transient > 0.4
        
        # 1. Beat phase alignment (CRITICAL - especially for drums)
        phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
        
        # ULTRA-STRICT tolerance for drums
        if has_drums:
            # For drums, be EXTREMELY strict - any misalignment is very audible
            tolerance = BEAT_PHASE_ACCEPTABLE * 0.5  # Even stricter
            excellent_threshold = BEAT_PHASE_EXCELLENT * 0.7
        else:
            tolerance = compute_adaptive_rhythm_tolerance(ctx.bpm)
            excellent_threshold = BEAT_PHASE_EXCELLENT
        
        # ULTRA-STRICT scoring - even tiny phase differences are penalized
        if phase_diff > tolerance:
            phase_score = 0.05  # Almost reject
        elif phase_diff > BEAT_PHASE_GOOD:
            phase_score = 0.25  # Poor
        elif phase_diff > excellent_threshold:
            phase_score = 0.6  # Acceptable
        else:
            # Excellent alignment - linear scale
            phase_score = 1.0 - phase_diff * 12  # Stricter scaling
        
        # BIG bonus for being exactly on beat (especially important for drums)
        on_beat_start = ctx.beat_phase_start < BEAT_PHASE_ON_BEAT_THRESHOLD or ctx.beat_phase_start > (1.0 - BEAT_PHASE_ON_BEAT_THRESHOLD)
        on_beat_end = ctx.beat_phase_end < BEAT_PHASE_ON_BEAT_THRESHOLD or ctx.beat_phase_end > (1.0 - BEAT_PHASE_ON_BEAT_THRESHOLD)
        if on_beat_start and on_beat_end:
            bonus = 0.25 if has_drums else 0.18  # Bigger bonus for drums
            phase_score = min(1.0, phase_score + bonus)
        elif on_beat_start or on_beat_end:
            # Partial bonus
            bonus = 0.1 if has_drums else 0.08
            phase_score = min(1.0, phase_score + bonus)
        
        # 2. Sub-beat phase alignment (16th note precision) - STRICTER
        sub_diff = circular_distance(ctx.sub_phase_start, ctx.sub_phase_end)
        
        if sub_diff > SUB_BEAT_GOOD:
            sub_score = 0.1  # Very poor
        elif sub_diff > SUB_BEAT_EXCELLENT:
            sub_score = 0.4  # Poor
        else:
            sub_score = 1.0 - sub_diff * 12  # Stricter scaling
        
        # 3. Phase coherence using Fourier representation
        phase_coherence = compute_phase_coherence(
            ctx.beat_phase_start, ctx.beat_phase_end,
            ctx.sub_phase_start, ctx.sub_phase_end
        )
        
        # 4. Loop duration alignment (should be multiple of beats/bars)
        duration_score = score_duration_alignment(
            ctx.loop_duration_frames, ctx.beat_length, ctx.bar_length
        )
        
        # 5. Tempo consistency (NEW) - check if loop maintains tempo
        tempo_score = self._check_tempo_consistency(ctx)
        
        # 6. Beat pattern continuity (NEW) - check if beat pattern continues smoothly
        pattern_score = self._check_beat_pattern_continuity(ctx)
        
        # Weighted combination (phase is most important, but add new metrics)
        score = (
            phase_score * 0.35 +      # Slightly reduced but still dominant
            sub_score * 0.20 +       # Reduced slightly
            phase_coherence * 0.15 +  # Reduced
            duration_score * 0.10 +  # Reduced
            tempo_score * 0.10 +     # NEW
            pattern_score * 0.10     # NEW
        )
        
        return max(0.0, min(1.0, score))
    
    def _check_tempo_consistency(self, ctx: TransitionContext) -> float:
        """Check if loop maintains consistent tempo."""
        # For now, assume tempo is consistent if phase alignment is good
        # This is a placeholder - could be enhanced with actual tempo tracking
        phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
        
        # If phase is well-aligned, tempo is likely consistent
        if phase_diff < BEAT_PHASE_EXCELLENT:
            return 1.0
        elif phase_diff < BEAT_PHASE_GOOD:
            return 0.7
        else:
            return 0.4
    
    def _check_beat_pattern_continuity(self, ctx: TransitionContext) -> float:
        """Check if beat pattern continues smoothly across transition."""
        # Check if we're transitioning at similar points in the beat pattern
        # This helps ensure groove continuity
        
        # If both points are on beat, pattern continues well
        on_beat_start = ctx.beat_phase_start < BEAT_PHASE_ON_BEAT_THRESHOLD or ctx.beat_phase_start > (1.0 - BEAT_PHASE_ON_BEAT_THRESHOLD)
        on_beat_end = ctx.beat_phase_end < BEAT_PHASE_ON_BEAT_THRESHOLD or ctx.beat_phase_end > (1.0 - BEAT_PHASE_ON_BEAT_THRESHOLD)
        
        if on_beat_start and on_beat_end:
            return 1.0
        elif on_beat_start or on_beat_end:
            return 0.7
        else:
            # Check phase difference
            phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
            if phase_diff < BEAT_PHASE_GOOD:
                return 0.6
            else:
                return 0.3
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
        sub_diff = circular_distance(ctx.sub_phase_start, ctx.sub_phase_end)
        
        return (
            f"Rhythm: {score:.2f} | "
            f"Beat phase diff: {phase_diff:.3f} | "
            f"Sub-beat diff: {sub_diff:.3f} | "
            f"Tempo: {ctx.bpm:.0f} BPM"
        )


@njit(cache=True, fastmath=True)
def compute_phase_coherence(
    phase1: float, phase2: float,
    sub1: float, sub2: float
) -> float:
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
    
    # Combine (beat more important)
    return (beat_coherence * 0.65 + sub_coherence * 0.35 + 1.0) / 2.0


def score_duration_alignment(
    duration_frames: int,
    beat_length: int,
    bar_length: int
) -> float:
    """Score how well loop duration aligns with musical structure."""
    if beat_length <= 0 or bar_length <= 0:
        return 0.5
    
    # Ideal durations in bars
    ideal_bars = [4, 8, 12, 16, 24, 32, 64]
    
    bars = duration_frames / bar_length
    
    best_match = 0.0
    for target in ideal_bars:
        for mult in [0.5, 1.0, 2.0]:
            deviation = abs(bars - target * mult) / (target * mult * 0.2 + 0.01)
            match = max(0, 1.0 - deviation)
            best_match = max(best_match, match)
    
    # Also check beat alignment
    start_off = duration_frames % beat_length
    beat_align = 1.0 - (start_off / beat_length) if beat_length > 0 else 0.5
    
    return best_match * 0.7 + beat_align * 0.3

