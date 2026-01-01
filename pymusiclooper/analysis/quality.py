"""
Loop Quality Metrics - Comprehensive Quality Assessment.

Provides detailed quality metrics for loop evaluation:
- Melodicity index
- Seamlessness score
- Rhythmic integrity
- Harmonic coherence
- Perceptual quality
- Technical quality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymusiclooper.analysis.features import Features
    from pymusiclooper.analysis.candidates import LoopPair


@dataclass
class LoopQualityMetrics:
    """Comprehensive quality metrics for a loop."""
    
    # Overall scores (0-1)
    overall_score: float
    
    # Melodic quality
    melodicity: float           # How melodic is the content
    melodic_continuity: float   # Does melody flow across loop
    pitch_stability: float      # Stable pitches at transition
    
    # Harmonic quality
    harmonic_coherence: float   # Key/chord consistency
    tonal_tension: float        # Musical tension level
    consonance: float           # Harmonic pleasantness
    
    # Rhythmic quality
    rhythm_integrity: float     # Beat alignment
    phase_alignment: float      # Sub-beat accuracy
    groove_continuity: float    # Rhythm pattern flows
    
    # Timbral quality
    timbral_match: float        # Spectral similarity
    spectral_stability: float   # Low spectral change
    brightness_match: float     # Spectral centroid similarity
    
    # Perceptual quality
    masking_effectiveness: float  # How well transition is masked
    loudness_continuity: float    # Energy flow
    imperceptibility: float       # Predicted audibility of cut
    
    # Technical quality
    crossfade_quality: float    # Waveform compatibility
    zero_crossing_opt: float    # Zero-crossing alignment
    transient_avoidance: float  # Avoiding drum hits
    
    # Structure
    phrase_alignment: float     # Musical phrase structure
    duration_quality: float     # Loop length appropriateness
    
    # Confidence
    confidence: float           # Confidence in assessment
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'overall_score': self.overall_score,
            'melodicity': self.melodicity,
            'melodic_continuity': self.melodic_continuity,
            'pitch_stability': self.pitch_stability,
            'harmonic_coherence': self.harmonic_coherence,
            'tonal_tension': self.tonal_tension,
            'consonance': self.consonance,
            'rhythm_integrity': self.rhythm_integrity,
            'phase_alignment': self.phase_alignment,
            'groove_continuity': self.groove_continuity,
            'timbral_match': self.timbral_match,
            'spectral_stability': self.spectral_stability,
            'brightness_match': self.brightness_match,
            'masking_effectiveness': self.masking_effectiveness,
            'loudness_continuity': self.loudness_continuity,
            'imperceptibility': self.imperceptibility,
            'crossfade_quality': self.crossfade_quality,
            'zero_crossing_opt': self.zero_crossing_opt,
            'transient_avoidance': self.transient_avoidance,
            'phrase_alignment': self.phrase_alignment,
            'duration_quality': self.duration_quality,
            'confidence': self.confidence,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        grade = _score_to_grade(self.overall_score)
        
        lines = [
            f"Overall Quality: {self.overall_score:.1%} ({grade})",
            "",
            f"Melodic:    {self.melodicity:.1%} melodicity, {self.melodic_continuity:.1%} continuity",
            f"Harmonic:   {self.harmonic_coherence:.1%} coherence, {self.consonance:.1%} consonance",
            f"Rhythmic:   {self.rhythm_integrity:.1%} integrity, {self.phase_alignment:.1%} phase",
            f"Timbral:    {self.timbral_match:.1%} match, {self.spectral_stability:.1%} stability",
            f"Perceptual: {self.imperceptibility:.1%} imperceptibility",
            f"Technical:  {self.crossfade_quality:.1%} crossfade, {self.transient_avoidance:.1%} transient avoid",
            "",
            f"Confidence: {self.confidence:.1%}",
        ]
        
        return "\n".join(lines)


def _score_to_grade(score: float) -> str:
    """Convert score to letter grade."""
    if score >= 0.9:
        return "A+"
    elif score >= 0.85:
        return "A"
    elif score >= 0.8:
        return "A-"
    elif score >= 0.75:
        return "B+"
    elif score >= 0.7:
        return "B"
    elif score >= 0.65:
        return "B-"
    elif score >= 0.6:
        return "C+"
    elif score >= 0.55:
        return "C"
    elif score >= 0.5:
        return "C-"
    elif score >= 0.4:
        return "D"
    else:
        return "F"


def compute_quality_metrics(
    feat: Features,
    loop_pair: LoopPair,
) -> LoopQualityMetrics:
    """
    Compute comprehensive quality metrics for a loop.
    
    This is the main function to get detailed quality analysis.
    """
    from pymusiclooper.analysis.experts.base import TransitionContext
    from pymusiclooper.analysis.experts.harmonic import HarmonicExpert, detect_key
    from pymusiclooper.analysis.experts.rhythm import RhythmExpert
    from pymusiclooper.analysis.experts.timbre import TimbreExpert
    from pymusiclooper.analysis.experts.psychoacoustic import PsychoacousticExpert
    from pymusiclooper.analysis.experts.neuroacoustic import (
        NeuroacousticExpert, consonance_model, roughness_model
    )
    from pymusiclooper.analysis.experts.dynamics import DynamicsExpert
    from pymusiclooper.analysis.experts.onset import OnsetExpert
    from pymusiclooper.analysis.experts.phrase import PhraseExpert
    from pymusiclooper.analysis.experts.melodic import (
        MelodicExpert, compute_melodicity_score, score_melodic_continuity
    )
    from pymusiclooper.analysis.experts.spectral import SpectralExpert
    from pymusiclooper.analysis.experts.crossfade import CrossfadeExpert
    
    # Create context
    ctx = TransitionContext.from_features(
        feat, loop_pair._loop_start_frame_idx, loop_pair._loop_end_frame_idx
    )
    
    # Initialize experts
    harmonic_expert = HarmonicExpert()
    rhythm_expert = RhythmExpert()
    timbre_expert = TimbreExpert()
    psycho_expert = PsychoacousticExpert()
    neuro_expert = NeuroacousticExpert()
    dynamics_expert = DynamicsExpert()
    onset_expert = OnsetExpert()
    phrase_expert = PhraseExpert()
    melodic_expert = MelodicExpert()
    spectral_expert = SpectralExpert()
    crossfade_expert = CrossfadeExpert()
    
    # Get expert scores
    harmonic_score = harmonic_expert.score(ctx)
    rhythm_score = rhythm_expert.score(ctx)
    timbre_score = timbre_expert.score(ctx)
    psycho_score = psycho_expert.score(ctx)
    neuro_score = neuro_expert.score(ctx)
    dynamics_score = dynamics_expert.score(ctx)
    onset_score = onset_expert.score(ctx)
    phrase_score = phrase_expert.score(ctx)
    melodic_score = melodic_expert.score(ctx)
    spectral_score = spectral_expert.score(ctx)
    crossfade_score = crossfade_expert.score(ctx)
    
    # Compute melodicity
    melodicity_curve = compute_melodicity_score(feat)
    start_melodicity = melodicity_curve[loop_pair._loop_start_frame_idx]
    end_melodicity = melodicity_curve[max(0, loop_pair._loop_end_frame_idx - 1)]
    avg_melodicity = float(np.mean(melodicity_curve[
        loop_pair._loop_start_frame_idx:loop_pair._loop_end_frame_idx
    ]))
    
    # Melodic continuity
    melodic_continuity = score_melodic_continuity(
        feat, loop_pair._loop_start_frame_idx, loop_pair._loop_end_frame_idx
    )
    
    # Pitch stability
    pitch_stability = _compute_pitch_stability(ctx)
    
    # Tonal tension
    from pymusiclooper.analysis.experts.neuroacoustic import harmonic_tension_model
    tonal_tension = harmonic_tension_model(ctx.chroma_start, ctx.chroma_end)
    
    # Consonance
    cons_start = consonance_model(ctx.chroma_start)
    cons_end = consonance_model(ctx.chroma_end)
    consonance = (cons_start + cons_end) / 2
    
    # Phase alignment (from rhythm)
    from pymusiclooper.analysis.experts.base import circular_distance
    phase_diff = circular_distance(ctx.beat_phase_start, ctx.beat_phase_end)
    phase_alignment = 1.0 - phase_diff * 2
    phase_alignment = max(0, phase_alignment)
    
    # Groove continuity
    groove_continuity = _compute_groove_continuity(feat, ctx)
    
    # Spectral stability
    spectral_stability = 1.0 - (ctx.flux_start + ctx.flux_end) / 2
    
    # Brightness match
    brightness_match = 1.0 - abs(ctx.centroid_start - ctx.centroid_end) * 5
    brightness_match = max(0, brightness_match)
    
    # Masking effectiveness
    masking = (ctx.masking_start + ctx.masking_end) / 2
    
    # Loudness continuity
    loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
    loudness_continuity = max(0, 1.0 - loudness_diff * 3)
    
    # Imperceptibility (combined perceptual)
    imperceptibility = psycho_score * 0.5 + masking * 0.3 + (1 - tonal_tension) * 0.2
    
    # Zero-crossing optimization
    zero_crossing_opt = 0.8  # Assumed good if we got here
    
    # Transient avoidance
    trans_max = max(ctx.transient_start, ctx.transient_end)
    transient_avoidance = 1.0 - trans_max
    
    # Duration quality
    duration_quality = _compute_duration_quality(
        loop_pair._loop_end_frame_idx - loop_pair._loop_start_frame_idx,
        feat.bar_length, feat.beat_length
    )
    
    # Confidence based on feature quality
    confidence = _compute_confidence(feat, loop_pair)
    
    # Overall score (weighted combination)
    overall = (
        melodic_score * 0.12 +
        harmonic_score * 0.12 +
        rhythm_score * 0.18 +
        timbre_score * 0.08 +
        psycho_score * 0.10 +
        neuro_score * 0.08 +
        onset_score * 0.12 +
        phrase_score * 0.05 +
        spectral_score * 0.05 +
        crossfade_score * 0.10
    )
    
    return LoopQualityMetrics(
        overall_score=overall,
        melodicity=avg_melodicity,
        melodic_continuity=melodic_continuity,
        pitch_stability=pitch_stability,
        harmonic_coherence=harmonic_score,
        tonal_tension=1.0 - tonal_tension,  # Invert so high = good
        consonance=consonance,
        rhythm_integrity=rhythm_score,
        phase_alignment=phase_alignment,
        groove_continuity=groove_continuity,
        timbral_match=timbre_score,
        spectral_stability=spectral_stability,
        brightness_match=brightness_match,
        masking_effectiveness=masking,
        loudness_continuity=loudness_continuity,
        imperceptibility=imperceptibility,
        crossfade_quality=crossfade_score,
        zero_crossing_opt=zero_crossing_opt,
        transient_avoidance=transient_avoidance,
        phrase_alignment=phrase_score,
        duration_quality=duration_quality,
        confidence=confidence,
    )


def _compute_pitch_stability(ctx) -> float:
    """Compute pitch stability at transition."""
    # Common prominent pitches
    threshold = 0.3
    common = 0.0
    total = 0.0
    
    for i in range(12):
        if ctx.chroma_start[i] > threshold or ctx.chroma_end[i] > threshold:
            total += 1
            if ctx.chroma_start[i] > threshold and ctx.chroma_end[i] > threshold:
                common += 1
    
    if total == 0:
        return 0.5
    
    return common / total


def _compute_groove_continuity(feat, ctx) -> float:
    """Compute groove/rhythm pattern continuity."""
    # Compare onset patterns around transition
    window = min(20, ctx.start_frame, feat.n_frames - ctx.end_frame)
    if window < 5:
        return 0.5
    
    pre_onset = feat.onset_env[max(0, ctx.end_frame - window):ctx.end_frame]
    post_onset = feat.onset_env[ctx.start_frame:min(ctx.start_frame + window, feat.n_frames)]
    
    min_len = min(len(pre_onset), len(post_onset))
    if min_len < 3:
        return 0.5
    
    pre_std = np.std(pre_onset[:min_len])
    post_std = np.std(post_onset[:min_len])
    
    if pre_std < 1e-6 or post_std < 1e-6:
        return 0.5
    
    corr = np.corrcoef(pre_onset[:min_len], post_onset[:min_len])[0, 1]
    return (corr + 1) / 2 if not np.isnan(corr) else 0.5


def _compute_duration_quality(duration_frames: int, bar_length: int, beat_length: int) -> float:
    """Score loop duration quality."""
    if bar_length <= 0:
        return 0.5
    
    bars = duration_frames / bar_length
    
    # Ideal bar counts
    ideal = [4, 8, 16, 32, 64]
    
    best = 0.0
    for target in ideal:
        deviation = abs(bars - target) / target
        score = max(0, 1.0 - deviation)
        best = max(best, score)
    
    # Power of 2 bonus
    bars_int = round(bars)
    is_power_of_2 = bars_int > 0 and (bars_int & (bars_int - 1)) == 0
    if is_power_of_2:
        best = min(1.0, best + 0.1)
    
    return best


def _compute_confidence(feat, loop_pair) -> float:
    """Compute confidence in quality assessment."""
    # Based on feature quality indicators
    
    # Duration confidence (very short = low confidence)
    duration = loop_pair._loop_end_frame_idx - loop_pair._loop_start_frame_idx
    duration_conf = min(1.0, duration / (feat.bar_length * 4)) if feat.bar_length > 0 else 0.5
    
    # Beat detection confidence
    beat_conf = min(1.0, len(feat.beats) / 20)
    
    # RMS signal level (very quiet = low confidence)
    avg_rms = np.mean(feat.rms[
        loop_pair._loop_start_frame_idx:loop_pair._loop_end_frame_idx
    ])
    rms_conf = min(1.0, avg_rms * 3)
    
    return (duration_conf + beat_conf + rms_conf) / 3


def compute_segment_melodicity(
    feat: Features,
    start_frame: int,
    end_frame: int,
) -> dict:
    """
    Compute detailed melodicity analysis for a segment.
    
    Returns:
        Dictionary with melodicity metrics.
    """
    from pymusiclooper.analysis.experts.melodic import compute_melodicity_score
    
    melodicity_curve = compute_melodicity_score(feat)
    segment_melodicity = melodicity_curve[start_frame:end_frame]
    
    return {
        'mean_melodicity': float(np.mean(segment_melodicity)),
        'max_melodicity': float(np.max(segment_melodicity)),
        'min_melodicity': float(np.min(segment_melodicity)),
        'melodicity_variance': float(np.var(segment_melodicity)),
        'high_melodicity_ratio': float(np.mean(segment_melodicity > 0.6)),
        'melodicity_curve': segment_melodicity,
    }


def rank_loops_by_quality(
    feat: Features,
    loops: list[LoopPair],
    criterion: str = 'overall',
) -> list[tuple[LoopPair, LoopQualityMetrics]]:
    """
    Rank loops by quality metrics.
    
    Args:
        feat: Audio features
        loops: List of loop pairs
        criterion: Ranking criterion ('overall', 'melodicity', 'rhythm', 'seamless')
    
    Returns:
        List of (loop, metrics) tuples sorted by criterion
    """
    results = []
    
    for loop in loops:
        metrics = compute_quality_metrics(feat, loop)
        results.append((loop, metrics))
    
    # Sort by criterion
    if criterion == 'melodicity':
        key = lambda x: x[1].melodicity * 0.5 + x[1].melodic_continuity * 0.5
    elif criterion == 'rhythm':
        key = lambda x: x[1].rhythm_integrity
    elif criterion == 'seamless':
        key = lambda x: x[1].imperceptibility
    else:  # overall
        key = lambda x: x[1].overall_score
    
    results.sort(key=key, reverse=True)
    
    return results

