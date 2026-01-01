"""
Timbre Expert - Spectral and Timbral Continuity.

Evaluates:
- MFCC similarity (spectral envelope)
- Spectral centroid continuity (brightness)
- Spectral bandwidth matching
- Spectral contrast consistency
- Formant-like features
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, cosine_similarity


class TimbreExpert(Expert):
    """Expert for timbral/spectral continuity."""
    
    name = "timbre"
    weight = 0.18  # Increased weight - timbre mismatch is very audible
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate timbral continuity at transition point."""
        
        # 1. MFCC similarity (spectral envelope match)
        mfcc_sim = cosine_similarity(ctx.mfcc_start, ctx.mfcc_end)
        
        # 2. MFCC context similarity (more stable)
        mfcc_context_sim = cosine_similarity(ctx.mfcc_context_start, ctx.mfcc_context_end)
        
        # 3. Spectral centroid match (brightness)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        centroid_score = max(0, 1.0 - centroid_diff * 5)
        
        # 4. Bandwidth match (spectral spread)
        bw_diff = abs(ctx.bandwidth_start - ctx.bandwidth_end)
        bw_score = max(0, 1.0 - bw_diff * 5)
        
        # 5. Flatness match (noise vs tonal)
        flat_diff = abs(ctx.flatness_start - ctx.flatness_end)
        flat_score = max(0, 1.0 - flat_diff * 3)
        
        # 6. Rolloff match (spectral tail)
        rolloff_diff = abs(ctx.rolloff_start - ctx.rolloff_end)
        rolloff_score = max(0, 1.0 - rolloff_diff * 4)
        
        # 7. Low flux at transition (stable timbre)
        avg_flux = (ctx.flux_start + ctx.flux_end) / 2
        flux_stability = 1.0 - avg_flux
        
        # Weighted combination
        score = (
            mfcc_sim * 0.25 +
            mfcc_context_sim * 0.20 +
            centroid_score * 0.15 +
            bw_score * 0.10 +
            flat_score * 0.10 +
            rolloff_score * 0.10 +
            flux_stability * 0.10
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        mfcc_sim = cosine_similarity(ctx.mfcc_start, ctx.mfcc_end)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        
        return (
            f"Timbre: {score:.2f} | "
            f"MFCC sim: {mfcc_sim:.2f} | "
            f"Centroid diff: {centroid_diff:.3f} | "
            f"Flux: {(ctx.flux_start + ctx.flux_end)/2:.2f}"
        )

