"""
Dynamics Expert - Energy and Loudness Flow.

Evaluates:
- RMS energy continuity
- Loudness matching
- Dynamic range consistency
- Envelope matching
- Compression-aware analysis
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class DynamicsExpert(Expert):
    """Expert for dynamic/energy continuity at transitions."""
    
    name = "dynamics"
    weight = 0.10
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate energy/loudness continuity at transition."""
        
        # 1. RMS energy match
        rms_diff = abs(ctx.rms_start - ctx.rms_end)
        rms_score = max(0, 1.0 - rms_diff * 4)
        
        # 2. RMS ratio (should be close to 1)
        rms_min = min(ctx.rms_start, ctx.rms_end) + 1e-10
        rms_max = max(ctx.rms_start, ctx.rms_end) + 1e-10
        rms_ratio = rms_min / rms_max
        ratio_score = rms_ratio ** 0.5  # Less harsh penalty
        
        # 3. Perceptual loudness match
        loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
        loudness_score = max(0, 1.0 - loudness_diff * 3)
        
        # 4. Energy level appropriateness
        # Very low energy = silence, hard to mask transitions
        avg_rms = (ctx.rms_start + ctx.rms_end) / 2
        energy_level_score = 0.3 + avg_rms * 0.7  # Some energy is good
        
        # 5. Flatness match (related to dynamic character)
        # High flatness = noise-like = dynamics less critical
        flat_avg = (ctx.flatness_start + ctx.flatness_end) / 2
        flat_diff = abs(ctx.flatness_start - ctx.flatness_end)
        flatness_score = (1.0 - flat_diff) * (0.8 + flat_avg * 0.2)
        
        # Combine scores
        score = (
            rms_score * 0.30 +
            ratio_score * 0.20 +
            loudness_score * 0.25 +
            energy_level_score * 0.15 +
            flatness_score * 0.10
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        rms_diff = abs(ctx.rms_start - ctx.rms_end)
        loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
        
        return (
            f"Dynamics: {score:.2f} | "
            f"RMS diff: {rms_diff:.3f} | "
            f"Loudness diff: {loudness_diff:.3f} | "
            f"Avg RMS: {(ctx.rms_start + ctx.rms_end)/2:.2f}"
        )

