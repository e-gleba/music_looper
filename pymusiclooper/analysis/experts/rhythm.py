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


class RhythmExpert(Expert):
    """Expert for rhythmic alignment and timing precision."""
    
    name = "rhythm"
    weight = 0.22  # High weight - rhythm is critical for seamless loops
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate rhythmic alignment at transition point."""
        
        # 1. Beat phase alignment (CRITICAL)
        phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
        
        # Strict scoring - even small phase differences are bad
        if phase_diff > 0.15:
            phase_score = 0.1
        elif phase_diff > 0.08:
            phase_score = 0.4
        elif phase_diff > 0.04:
            phase_score = 0.7
        else:
            phase_score = 1.0 - phase_diff * 8
        
        # Bonus for being exactly on beat
        on_beat_start = ctx.beat_phase_start < 0.06 or ctx.beat_phase_start > 0.94
        on_beat_end = ctx.beat_phase_end < 0.06 or ctx.beat_phase_end > 0.94
        if on_beat_start and on_beat_end:
            phase_score = min(1.0, phase_score + 0.15)
        
        # 2. Sub-beat phase alignment (16th note precision)
        sub_diff = circular_distance(ctx.sub_phase_start, ctx.sub_phase_end)
        
        if sub_diff > 0.12:
            sub_score = 0.15
        elif sub_diff > 0.06:
            sub_score = 0.5
        else:
            sub_score = 1.0 - sub_diff * 10
        
        # 3. Phase coherence using Fourier representation
        phase_coherence = compute_phase_coherence(
            ctx.beat_phase_start, ctx.beat_phase_end,
            ctx.sub_phase_start, ctx.sub_phase_end
        )
        
        # 4. Loop duration alignment (should be multiple of beats/bars)
        duration_score = score_duration_alignment(
            ctx.loop_duration_frames, ctx.beat_length, ctx.bar_length
        )
        
        # Weighted combination (phase is most important)
        score = (
            phase_score * 0.40 +
            sub_score * 0.25 +
            phase_coherence * 0.20 +
            duration_score * 0.15
        )
        
        return max(0.0, min(1.0, score))
    
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

