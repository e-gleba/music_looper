"""
Phrase Expert - Musical Structure and Phrase Boundaries.

Evaluates:
- Loop alignment with phrase structure (4, 8, 16 bars)
- Downbeat alignment
- Musical section boundaries
- Motif completion
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class PhraseExpert(Expert):
    """Expert for musical phrase/structure alignment."""
    
    name = "phrase"
    weight = 0.08
    
    def score(self, ctx: TransitionContext) -> float:
        """Evaluate structural alignment of the loop."""
        
        if ctx.bar_length <= 0 or ctx.beat_length <= 0:
            return 0.5  # Can't evaluate structure
        
        # 1. Bar alignment score
        # Loop length should be multiple of bars
        bars = ctx.loop_duration_frames / ctx.bar_length
        ideal_bars = [4, 8, 12, 16, 24, 32, 64]
        
        bar_score = 0.0
        for target in ideal_bars:
            for mult in [0.5, 1.0, 2.0]:
                deviation = abs(bars - target * mult) / (target * mult * 0.2 + 0.01)
                match = max(0, 1.0 - deviation)
                bar_score = max(bar_score, match)
        
        # 2. Beat alignment at boundaries
        start_beat_offset = ctx.start_frame % ctx.beat_length
        end_beat_offset = ctx.end_frame % ctx.beat_length
        
        start_on_beat = start_beat_offset < ctx.beat_length * 0.1
        end_on_beat = end_beat_offset < ctx.beat_length * 0.1
        
        beat_align_score = (float(start_on_beat) + float(end_on_beat)) / 2
        
        # 3. Downbeat alignment (even stronger boundaries)
        start_bar_offset = ctx.start_frame % ctx.bar_length
        end_bar_offset = ctx.end_frame % ctx.bar_length
        
        start_on_downbeat = start_bar_offset < ctx.beat_length * 0.15
        end_on_downbeat = end_bar_offset < ctx.beat_length * 0.15
        
        downbeat_score = (float(start_on_downbeat) + float(end_on_downbeat)) / 2
        
        # 4. Loop coverage (relative to song)
        # Very short loops are often incomplete phrases
        coverage = ctx.loop_duration_frames / ctx.n_frames
        if coverage < 0.1:
            coverage_penalty = 0.3
        elif coverage < 0.2:
            coverage_penalty = 0.7
        else:
            coverage_penalty = 1.0
        
        # 5. Power of 2 bar count (musical preference)
        bars_rounded = round(bars)
        is_power_of_2 = bars_rounded > 0 and (bars_rounded & (bars_rounded - 1)) == 0
        power_of_2_bonus = 0.15 if is_power_of_2 else 0.0
        
        # Combine scores
        score = (
            bar_score * 0.35 +
            beat_align_score * 0.20 +
            downbeat_score * 0.20 +
            power_of_2_bonus
        ) * coverage_penalty
        
        return max(0.0, min(1.0, score))
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        bars = ctx.loop_duration_frames / ctx.bar_length if ctx.bar_length > 0 else 0
        coverage = ctx.loop_duration_frames / ctx.n_frames * 100
        
        return (
            f"Phrase: {score:.2f} | "
            f"Bars: {bars:.1f} | "
            f"Coverage: {coverage:.1f}% | "
            f"BPM: {ctx.bpm:.0f}"
        )

