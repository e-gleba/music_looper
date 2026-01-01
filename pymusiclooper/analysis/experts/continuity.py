"""
Continuity Expert - Overall Seamlessness and Flow.

Evaluates:
- Overall transition smoothness
- Multi-dimensional continuity
- Seamlessness score
- Organic flow preservation
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class ContinuityExpert(Expert):
    """
    Expert for overall continuity and seamlessness.
    
    Provides a holistic view of transition quality by combining
    multiple continuity aspects into a unified seamlessness score.
    """
    
    name = "continuity"
    weight = 0.12  # High weight - overall continuity is critical
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate overall continuity and seamlessness."""
        
        # 1. Harmonic continuity (chroma similarity)
        chroma_sim = np.dot(ctx.chroma_start, ctx.chroma_end) / (
            np.linalg.norm(ctx.chroma_start) * np.linalg.norm(ctx.chroma_end) + 1e-10
        )
        
        # 2. Timbral continuity (MFCC similarity)
        mfcc_sim = np.dot(ctx.mfcc_start, ctx.mfcc_end) / (
            np.linalg.norm(ctx.mfcc_start) * np.linalg.norm(ctx.mfcc_end) + 1e-10
        )
        
        # 3. Rhythmic continuity (phase alignment)
        from pymusiclooper.analysis.experts.base import circular_distance
        phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
        rhythm_continuity = 1.0 - phase_diff * 2
        
        # 4. Dynamic continuity (energy flow)
        energy_diff = abs(ctx.rms_start - ctx.rms_end)
        dynamic_continuity = 1.0 - min(energy_diff * 2, 1.0)
        
        # 5. Spectral continuity (centroid, bandwidth)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        bandwidth_diff = abs(ctx.bandwidth_start - ctx.bandwidth_end)
        spectral_continuity = 1.0 - (centroid_diff * 2.5 + bandwidth_diff * 2.5) / 2
        
        # 6. Transient continuity (avoid cutting at transients)
        transient_max = max(ctx.transient_start, ctx.transient_end)
        transient_continuity = 1.0 - transient_max
        
        # 7. Flux continuity (spectral change)
        flux_avg = (ctx.flux_start + ctx.flux_end) / 2
        flux_continuity = 1.0 - flux_avg
        
        # Weighted combination - all aspects matter for true seamlessness
        score = (
            chroma_sim * 0.18 +
            mfcc_sim * 0.18 +
            rhythm_continuity * 0.20 +  # Rhythm is critical
            dynamic_continuity * 0.15 +
            spectral_continuity * 0.12 +
            transient_continuity * 0.10 +
            flux_continuity * 0.07
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        from pymusiclooper.analysis.experts.base import circular_distance
        phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
        return f"Continuity: {score:.2f} | Phase diff: {phase_diff:.3f}"

