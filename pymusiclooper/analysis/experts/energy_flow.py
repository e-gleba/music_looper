"""
Energy Flow Expert - Temporal Energy Dynamics.

Evaluates:
- Energy envelope continuity
- Dynamic flow smoothness
- Energy gradient matching
- Momentum preservation
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class EnergyFlowExpert(Expert):
    """
    Expert for energy flow and dynamic continuity.
    
    Analyzes how energy flows across the transition:
    - Energy envelope matching
    - Dynamic gradients
    - Momentum preservation
    """
    
    name = "energy_flow"
    weight = 0.11
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate energy flow continuity."""
        
        # 1. Energy level matching
        energy_diff = abs(ctx.rms_start - ctx.rms_end)
        energy_match = 1.0 - min(energy_diff * 2, 1.0)
        
        # 2. Energy gradient matching
        # Check if energy is changing in similar ways at both ends
        gradient_score = compute_energy_gradient_match(ctx)
        
        # 3. Momentum preservation
        # Energy should flow naturally (no sudden stops/starts)
        momentum_score = compute_momentum_preservation(ctx)
        
        # 4. Dynamic contour matching
        # The shape of the energy curve should match
        contour_score = compute_dynamic_contour_match(ctx)
        
        # 5. Loudness continuity
        loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
        loudness_match = 1.0 - min(loudness_diff * 3, 1.0)
        
        # Weighted combination
        score = (
            energy_match * 0.25 +
            gradient_score * 0.25 +
            momentum_score * 0.20 +
            contour_score * 0.15 +
            loudness_match * 0.15
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        energy_diff = abs(ctx.rms_start - ctx.rms_end)
        return f"Energy Flow: {score:.2f} | RMS diff: {energy_diff:.3f}"


def compute_energy_gradient_match(ctx: TransitionContext) -> float:
    """
    Compute energy gradient matching.
    
    Checks if energy is changing in similar ways at both transition points.
    """
    # This would need access to RMS history, simplified here
    # In practice, we'd compare gradients from context windows
    
    # For now, use RMS difference as proxy
    rms_diff = abs(ctx.rms_start - ctx.rms_end)
    
    # If both are similar, gradient is likely similar
    if rms_diff < 0.1:
        return 0.9
    elif rms_diff < 0.2:
        return 0.7
    elif rms_diff < 0.3:
        return 0.5
    else:
        return 0.3


def compute_momentum_preservation(ctx: TransitionContext) -> float:
    """
    Compute momentum preservation.
    
    Checks if energy momentum is preserved (no sudden stops/starts).
    """
    # Ideal: energy at end should flow naturally into energy at start
    # If end is high and start is low, that's a momentum break
    
    energy_ratio = ctx.rms_end / (ctx.rms_start + 1e-10)
    
    # Ideal ratio is close to 1.0 (smooth flow)
    if 0.8 <= energy_ratio <= 1.2:
        return 1.0
    elif 0.6 <= energy_ratio <= 1.5:
        return 0.7
    elif 0.4 <= energy_ratio <= 2.0:
        return 0.4
    else:
        return 0.2  # Large momentum break


def compute_dynamic_contour_match(ctx: TransitionContext) -> float:
    """
    Compute dynamic contour matching.
    
    Checks if the shape of the energy envelope matches.
    """
    # Simplified: compare RMS and loudness together
    rms_sim = 1.0 - abs(ctx.rms_start - ctx.rms_end)
    loud_sim = 1.0 - abs(ctx.loudness_start - ctx.loudness_end)
    
    # Also check if both are in similar dynamic ranges
    rms_avg = (ctx.rms_start + ctx.rms_end) / 2
    rms_range_match = 1.0 if rms_avg > 0.1 else 0.7  # Prefer non-silent regions
    
    return (rms_sim * 0.4 + loud_sim * 0.4 + rms_range_match * 0.2)

