"""
Transition Effects Expert - Evaluates and Optimizes Transition Effects.

Analyzes transition quality and recommends optimal effects:
- Rhythm-masking effectiveness
- Harmonic alignment quality
- Reverb/delay appropriateness
- Overall transition smoothness
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class TransitionEffectsExpert(Expert):
    """Expert for evaluating and optimizing transition effects."""
    
    name = "transition_effects"
    weight = 0.18  # High weight - effects are critical for smooth transitions
    
    def score(self, ctx: TransitionContext) -> float:
        """
        Evaluate transition quality and recommend effects.
        
        Analyzes:
        1. Rhythm continuity (needs masking?)
        2. Harmonic alignment (needs pitch correction?)
        3. Spectral smoothness (needs reverb/delay?)
        4. Energy flow (needs time-stretching?)
        """
        # 1. Rhythm continuity score
        rhythm_score = self._evaluate_rhythm_continuity(ctx)
        
        # 2. Harmonic alignment score
        harmonic_score = self._evaluate_harmonic_alignment(ctx)
        
        # 3. Spectral smoothness score
        spectral_score = self._evaluate_spectral_smoothness(ctx)
        
        # 4. Energy flow score
        energy_score = self._evaluate_energy_flow(ctx)
        
        # 5. Transient handling score
        transient_score = self._evaluate_transient_handling(ctx)
        
        # Combined score (weighted by importance)
        score = (
            rhythm_score * 0.30 +      # Most important - rhythm must be smooth
            harmonic_score * 0.25 +    # Important for melodic music
            spectral_score * 0.20 +    # Important for smoothness
            energy_score * 0.15 +      # Important for dynamics
            transient_score * 0.10     # Important for drums
        )
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_rhythm_continuity(self, ctx: TransitionContext) -> float:
        """Evaluate if rhythm continues smoothly across transition."""
        # Check beat phase alignment
        from pymusiclooper.analysis.experts.base import circular_distance
        from pymusiclooper.analysis.constants import BEAT_PHASE_ACCEPTABLE
        
        phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
        
        # Good alignment = high score
        if phase_diff < BEAT_PHASE_ACCEPTABLE * 0.5:
            return 1.0
        elif phase_diff < BEAT_PHASE_ACCEPTABLE:
            return 0.8
        elif phase_diff < BEAT_PHASE_ACCEPTABLE * 1.5:
            return 0.5
        else:
            return 0.2  # Poor - needs rhythm masking
    
    def _evaluate_harmonic_alignment(self, ctx: TransitionContext) -> float:
        """Evaluate if harmonics align well (needs pitch correction?)."""
        # Check chroma similarity
        chroma_sim = np.dot(ctx.chroma_start, ctx.chroma_end) / (
            np.linalg.norm(ctx.chroma_start) * np.linalg.norm(ctx.chroma_end) + 1e-10
        )
        
        # Check tonnetz similarity (harmonic relationships)
        tonnetz_sim = np.dot(ctx.tonnetz_start, ctx.tonnetz_end) / (
            np.linalg.norm(ctx.tonnetz_start) * np.linalg.norm(ctx.tonnetz_end) + 1e-10
        )
        
        # Combined harmonic score
        harmonic_score = (chroma_sim * 0.6 + tonnetz_sim * 0.4)
        
        return float(harmonic_score)
    
    def _evaluate_spectral_smoothness(self, ctx: TransitionContext) -> float:
        """Evaluate spectral smoothness (needs reverb/delay?)."""
        # Check spectral flux (low = smooth)
        avg_flux = (ctx.flux_start + ctx.flux_end) / 2
        
        # Check spectral centroid difference (similar = smooth)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        
        # Check bandwidth difference
        bandwidth_diff = abs(ctx.bandwidth_start - ctx.bandwidth_end)
        
        # Smooth = low flux, low differences
        flux_score = 1.0 - min(1.0, avg_flux * 2.0)
        centroid_score = 1.0 - min(1.0, centroid_diff * 3.0)
        bandwidth_score = 1.0 - min(1.0, bandwidth_diff * 2.0)
        
        return (flux_score * 0.4 + centroid_score * 0.35 + bandwidth_score * 0.25)
    
    def _evaluate_energy_flow(self, ctx: TransitionContext) -> float:
        """Evaluate energy flow continuity."""
        # Check RMS difference
        rms_diff = abs(ctx.rms_start - ctx.rms_end) / (max(ctx.rms_start, ctx.rms_end) + 1e-10)
        
        # Check loudness difference
        loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
        
        # Good flow = low differences
        rms_score = 1.0 - min(1.0, rms_diff)
        loudness_score = 1.0 - min(1.0, loudness_diff)
        
        return (rms_score * 0.6 + loudness_score * 0.4)
    
    def _evaluate_transient_handling(self, ctx: TransitionContext) -> float:
        """Evaluate transient handling (critical for drums)."""
        # Check if transients are present
        avg_transient = (ctx.transient_start + ctx.transient_end) / 2
        avg_onset = (ctx.onset_start + ctx.onset_end) / 2
        
        # If high transients, need careful handling
        if avg_transient > 0.5 or avg_onset > 0.5:
            # Check if we're avoiding cutting at transients
            # Low transient at cut point = good
            return 1.0 - min(1.0, avg_transient * 0.8)
        else:
            # Low transients = good for smooth transition
            return 1.0
    
    def recommend_effects(
        self,
        ctx: TransitionContext,
    ) -> dict:
        """
        Recommend specific effects based on transition analysis.
        
        Returns:
            dict with recommended effects and parameters
        """
        rhythm_score = self._evaluate_rhythm_continuity(ctx)
        harmonic_score = self._evaluate_harmonic_alignment(ctx)
        spectral_score = self._evaluate_spectral_smoothness(ctx)
        avg_transient = (ctx.transient_start + ctx.transient_end) / 2
        
        recommendations = {
            'use_rhythm_masking': rhythm_score < 0.6,
            'use_harmonic_alignment': harmonic_score < 0.7,
            'use_reverb_tail': spectral_score < 0.6 or avg_transient < 0.2,
            'use_delay_echo': rhythm_score > 0.6 and avg_transient > 0.3,
            'rhythm_gap_ms': 0.0,  # Will be detected during transition
            'reverb_length_ms': 50 if spectral_score < 0.5 else 30,
            'delay_ms': 20 if avg_transient > 0.4 else 15,
        }
        
        return recommendations
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        recommendations = self.recommend_effects(ctx)
        
        effects = []
        if recommendations['use_rhythm_masking']:
            effects.append("rhythm-mask")
        if recommendations['use_harmonic_alignment']:
            effects.append("harmonic")
        if recommendations['use_reverb_tail']:
            effects.append("reverb")
        if recommendations['use_delay_echo']:
            effects.append("delay")
        
        effects_str = ", ".join(effects) if effects else "none"
        
        return (
            f"Effects: {score:.2f} | "
            f"Recommended: {effects_str}"
        )

