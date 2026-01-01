"""
Resonance Expert - Formant and Resonance Analysis.

Evaluates:
- Formant matching (vocal/instrumental resonance)
- Spectral peak alignment
- Resonance frequency continuity
- Harmonic series alignment
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class ResonanceExpert(Expert):
    """
    Expert for resonance and formant analysis.
    
    Analyzes spectral resonances that create timbral character:
    - Formant frequencies (vocal/instrumental)
    - Spectral peak alignment
    - Resonance continuity
    """
    
    name = "resonance"
    weight = 0.09
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate resonance and formant continuity."""
        
        # 1. Spectral centroid matching (brightness/resonance)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        centroid_match = 1.0 - min(centroid_diff * 5, 1.0)
        
        # 2. Spectral rolloff matching (high-frequency content)
        rolloff_diff = abs(ctx.rolloff_start - ctx.rolloff_end)
        rolloff_match = 1.0 - min(rolloff_diff * 4, 1.0)
        
        # 3. Bandwidth matching (spectral spread)
        bandwidth_diff = abs(ctx.bandwidth_start - ctx.bandwidth_end)
        bandwidth_match = 1.0 - min(bandwidth_diff * 5, 1.0)
        
        # 4. Formant-like analysis using spectral peaks
        # Higher centroid + lower bandwidth = more formant-like
        formant_char_start = ctx.centroid_start * (1.0 - ctx.bandwidth_start)
        formant_char_end = ctx.centroid_end * (1.0 - ctx.bandwidth_end)
        formant_match = 1.0 - abs(formant_char_start - formant_char_end)
        
        # 5. Spectral stability (low flux = stable resonances)
        flux_avg = (ctx.flux_start + ctx.flux_end) / 2
        stability = 1.0 - flux_avg
        
        # Weighted combination
        score = (
            centroid_match * 0.25 +
            rolloff_match * 0.20 +
            bandwidth_match * 0.20 +
            formant_match * 0.20 +
            stability * 0.15
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        return f"Resonance: {score:.2f} | Centroid diff: {centroid_diff:.3f}"

