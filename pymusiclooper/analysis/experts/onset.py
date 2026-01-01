"""
Onset Expert - Transient and Attack Handling.

Evaluates:
- Transient avoidance (don't cut at drum hits)
- Onset continuity
- Attack envelope matching
- Percussive vs sustained content handling
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class OnsetExpert(Expert):
    """Expert for transient/onset handling at transitions."""
    
    name = "onset"
    weight = 0.15  # High weight - cutting at transients is very audible
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate transient handling at transition point."""
        
        # 1. Transient strength at cut points
        # CRITICAL: High transients at cut = very audible pop/click
        trans_max = max(ctx.transient_start, ctx.transient_end)
        
        if trans_max > 0.7:
            transient_score = 0.1  # Very bad
        elif trans_max > 0.5:
            transient_score = 0.3
        elif trans_max > 0.3:
            transient_score = 0.6
        else:
            transient_score = 0.8 + (0.3 - trans_max) * 0.6
        
        # 2. Onset peaks at cut points
        # Cutting exactly at an onset is very audible
        onset_max = max(ctx.onset_start, ctx.onset_end)
        
        if onset_max > 0.5:
            onset_score = 0.2  # Very bad to cut at onset
        else:
            onset_score = 1.0 - onset_max * 1.5
        
        # 3. Transient match (similar transient levels = smoother transition)
        trans_diff = abs(ctx.transient_start - ctx.transient_end)
        trans_match = max(0, 1.0 - trans_diff * 2)
        
        # 4. Spectral flux at cut (high flux = timbral change = audible)
        flux_max = max(ctx.flux_start, ctx.flux_end)
        flux_score = 1.0 - flux_max
        
        # 5. Combined attack character
        # If both endpoints have similar "attackiness", transition is smoother
        attack_char_start = ctx.transient_start * 0.6 + ctx.flux_start * 0.4
        attack_char_end = ctx.transient_end * 0.6 + ctx.flux_end * 0.4
        attack_match = 1.0 - abs(attack_char_start - attack_char_end)
        
        # Combine with heavy weight on avoiding transients at cut
        score = (
            transient_score * 0.35 +
            onset_score * 0.25 +
            trans_match * 0.15 +
            flux_score * 0.15 +
            attack_match * 0.10
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        trans_max = max(ctx.transient_start, ctx.transient_end)
        onset_max = max(ctx.onset_start, ctx.onset_end)
        
        return (
            f"Onset: {score:.2f} | "
            f"Transient: {trans_max:.2f} | "
            f"Onset peak: {onset_max:.2f} | "
            f"Flux: {max(ctx.flux_start, ctx.flux_end):.2f}"
        )

