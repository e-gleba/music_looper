"""
Microtiming Expert - Groove, Swing, and Micro-timing Analysis.

Evaluates:
- Groove continuity (swing, shuffle patterns)
- Micro-timing consistency
- Sub-beat rhythmic feel
- Temporal flow smoothness
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, circular_distance


class MicrotimingExpert(Expert):
    """
    Expert for micro-timing and groove analysis.
    
    Analyzes subtle timing variations that create musical feel:
    - Swing/shuffle patterns
    - Groove consistency
    - Sub-beat timing feel
    """
    
    name = "microtiming"
    weight = 0.10
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate micro-timing and groove continuity."""
        
        # 1. Sub-beat phase alignment (16th note feel)
        sub_phase_diff = circular_distance(ctx.sub_phase_start, ctx.sub_phase_end)
        
        # For groove, we want consistent micro-timing
        # Small differences are OK (creates feel), but large differences break groove
        if sub_phase_diff < 0.05:
            sub_score = 1.0  # Perfect alignment
        elif sub_phase_diff < 0.12:
            sub_score = 0.8  # Good - slight variation is OK
        elif sub_phase_diff < 0.25:
            sub_score = 0.5  # Moderate - noticeable but acceptable
        else:
            sub_score = 0.2  # Bad - breaks groove
        
        # 2. Groove pattern consistency
        # Check if the "feel" (swing/shuffle) is consistent
        groove_score = compute_groove_consistency(
            ctx.sub_phase_start, ctx.sub_phase_end, ctx.bpm
        )
        
        # 3. Temporal flow smoothness
        # Analyze how smoothly time flows across the transition
        flow_score = compute_temporal_flow(ctx)
        
        # 4. Beat subdivision alignment
        # Check alignment to common subdivisions (triplets, 16ths)
        subdivision_score = score_subdivision_alignment(
            ctx.sub_phase_start, ctx.sub_phase_end
        )
        
        # Weighted combination
        score = (
            sub_score * 0.30 +
            groove_score * 0.30 +
            flow_score * 0.25 +
            subdivision_score * 0.15
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        sub_diff = circular_distance(ctx.sub_phase_start, ctx.sub_phase_end)
        return f"Microtiming: {score:.2f} | Sub-beat diff: {sub_diff:.3f}"


@njit(cache=True, fastmath=True)
def compute_groove_consistency(
    phase1: float, phase2: float, bpm: float
) -> float:
    """
    Compute groove pattern consistency.
    
    Checks if the micro-timing "feel" is similar at both ends.
    """
    # Convert phases to "feel" vectors
    # For swing/shuffle, we look at the relationship between beats
    
    # Simple model: check if phases are in similar "zones"
    # Zone 0: early (0-0.25), Zone 1: on-time (0.25-0.75), Zone 2: late (0.75-1.0)
    zone1 = int(phase1 * 3) if phase1 < 1.0 else 2
    zone2 = int(phase2 * 3) if phase2 < 1.0 else 2
    
    if zone1 == zone2:
        return 1.0
    elif abs(zone1 - zone2) == 1:
        return 0.6  # Adjacent zones - similar feel
    else:
        return 0.2  # Opposite zones - different feel


def compute_temporal_flow(ctx: TransitionContext) -> float:
    """
    Analyze temporal flow smoothness.
    
    Checks how smoothly time flows across the transition,
    considering beat phase, sub-beat phase, and tempo consistency.
    """
    # Phase velocity (rate of change)
    beat_phase_vel = abs(ctx.beat_phase_end - ctx.beat_phase_start)
    sub_phase_vel = abs(ctx.sub_phase_end - ctx.sub_phase_start)
    
    # Normalize by loop duration
    duration_beats = ctx.loop_duration_frames / ctx.beat_length if ctx.beat_length > 0 else 1.0
    
    # Ideal: phases should align naturally after loop duration
    expected_beat_phase = (ctx.beat_phase_start + duration_beats) % 1.0
    expected_sub_phase = (ctx.sub_phase_start + duration_beats * 4) % 1.0
    
    beat_phase_error = circular_distance(expected_beat_phase, ctx.beat_phase_end)
    sub_phase_error = circular_distance(expected_sub_phase, ctx.sub_phase_end)
    
    # Lower error = smoother flow
    beat_flow = 1.0 - beat_phase_error * 2
    sub_flow = 1.0 - sub_phase_error * 2
    
    return (beat_flow * 0.6 + sub_flow * 0.4)


@njit(cache=True, fastmath=True)
def score_subdivision_alignment(phase1: float, phase2: float) -> float:
    """
    Score alignment to common beat subdivisions.
    
    Checks if both phases align to common subdivisions:
    - Triplets (0, 0.33, 0.67)
    - 16ths (0, 0.25, 0.5, 0.75)
    - 8ths (0, 0.5)
    """
    # Common subdivision points
    subdivisions = np.array([
        0.0, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0
    ], dtype=np.float32)
    
    # Find nearest subdivision for each phase
    dist1 = np.abs(subdivisions - phase1)
    dist2 = np.abs(subdivisions - phase2)
    
    nearest1 = subdivisions[np.argmin(dist1)]
    nearest2 = subdivisions[np.argmin(dist2)]
    
    # Score based on how close to subdivisions and if they match
    closeness1 = 1.0 - np.min(dist1) * 4
    closeness2 = 1.0 - np.min(dist2) * 4
    
    match_bonus = 0.3 if abs(nearest1 - nearest2) < 0.01 else 0.0
    
    return (closeness1 + closeness2) / 2 + match_bonus

