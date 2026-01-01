"""
Texture Expert - Spectral Texture and Density Analysis.

Evaluates:
- Spectral texture matching (dense vs sparse)
- Harmonic density consistency
- Spectral complexity matching
- Texture continuity
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext, cosine_similarity


class TextureExpert(Expert):
    """
    Expert for spectral texture analysis.
    
    Analyzes the "texture" of the sound:
    - Density (how many frequencies are active)
    - Complexity (spectral richness)
    - Sparsity vs density
    """
    
    name = "texture"
    weight = 0.08
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate texture continuity."""
        
        # 1. Spectral flatness matching (noise vs tonal)
        # Low flatness = tonal (sparse), high = noisy (dense)
        flatness_diff = abs(ctx.flatness_start - ctx.flatness_end)
        flatness_match = 1.0 - min(flatness_diff * 3, 1.0)
        
        # 2. Spectral bandwidth matching (spread = density indicator)
        bandwidth_diff = abs(ctx.bandwidth_start - ctx.bandwidth_end)
        bandwidth_match = 1.0 - min(bandwidth_diff * 5, 1.0)
        
        # 3. Chroma density (how many pitch classes are active)
        chroma_density_start = np.sum(ctx.chroma_start > 0.1) / 12.0
        chroma_density_end = np.sum(ctx.chroma_end > 0.1) / 12.0
        density_diff = abs(chroma_density_start - chroma_density_end)
        density_match = 1.0 - density_diff
        
        # 4. MFCC texture (spectral envelope shape = texture)
        mfcc_texture_sim = cosine_similarity(ctx.mfcc_start, ctx.mfcc_end)
        
        # 5. Spectral complexity (flux indicates complexity)
        flux_diff = abs(ctx.flux_start - ctx.flux_end)
        complexity_match = 1.0 - min(flux_diff * 2, 1.0)
        
        # Weighted combination
        score = (
            flatness_match * 0.20 +
            bandwidth_match * 0.20 +
            density_match * 0.20 +
            mfcc_texture_sim * 0.25 +
            complexity_match * 0.15
        )
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        flatness_diff = abs(ctx.flatness_start - ctx.flatness_end)
        return f"Texture: {score:.2f} | Flatness diff: {flatness_diff:.3f}"

