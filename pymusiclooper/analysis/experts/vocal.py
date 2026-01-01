"""
Vocal/Choral Analysis Expert - Word Boundary Detection.

Detects vocal pauses, word boundaries, and phrase endings to find
optimal loop points that avoid cutting mid-word or mid-phrase.
Especially important for choral music and vocal tracks.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.fft import rfft

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class VocalExpert(Expert):
    """Expert for detecting vocal pauses and word boundaries.
    
    CRITICAL: Prevents transitions where vocals appear/disappear.
    """
    
    name = "vocal"
    weight = 0.20  # Increased weight - critical for vocal music
    
    def score(self, ctx: TransitionContext) -> float:
        """
        Evaluate transition quality for vocal content.
        
        CRITICAL: Rejects transitions where vocals appear in one part but not the other.
        This prevents audible breaks when vocals start/stop.
        """
        # 1. DETECT VOCAL PRESENCE in both parts (CRITICAL)
        has_vocal_start = self._detect_vocal_presence(ctx, is_start=True)
        has_vocal_end = self._detect_vocal_presence(ctx, is_start=False)
        
        # HARD REJECTION: If vocals in one part but not the other
        if has_vocal_start != has_vocal_end:
            # One part has vocals, other doesn't - this creates audible break
            return 0.05  # Almost reject - very bad transition
        
        # If both have vocals or both don't, continue with normal scoring
        # 2. Energy-based pause detection
        # Low energy regions are likely pauses between words/phrases
        avg_energy = (ctx.rms_start + ctx.rms_end) / 2
        energy_score = 1.0 - min(1.0, avg_energy * 1.5)  # Prefer lower energy
        
        # 3. Spectral stability (low flux = stable vocal content)
        avg_flux = (ctx.flux_start + ctx.flux_end) / 2
        flux_score = 1.0 - min(1.0, avg_flux * 2.0)  # Prefer lower flux
        
        # 4. Onset absence (avoid cutting during vocal attacks)
        avg_onset = (ctx.onset_start + ctx.onset_end) / 2
        onset_score = 1.0 - min(1.0, avg_onset * 2.0)  # Prefer no onsets
        
        # 5. MFCC consistency (similar vocal timbre) - STRICTER for vocals
        mfcc_diff = np.linalg.norm(ctx.mfcc_start - ctx.mfcc_end)
        if has_vocal_start and has_vocal_end:
            # Both have vocals - need very similar timbre
            mfcc_score = 1.0 - min(1.0, mfcc_diff / 3.0)  # Stricter
        else:
            # No vocals - more lenient
            mfcc_score = 1.0 - min(1.0, mfcc_diff / 5.0)
        
        # 6. Spectral centroid stability (vocal formants should be similar)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        if has_vocal_start and has_vocal_end:
            # Both have vocals - need very similar formants
            centroid_score = 1.0 - min(1.0, centroid_diff * 4.0)  # Stricter
        else:
            centroid_score = 1.0 - min(1.0, centroid_diff * 3.0)
        
        # 7. Melodic continuity (for vocal lines)
        melodic_score = self._evaluate_melodic_continuity(ctx, has_vocal_start, has_vocal_end)
        
        # 8. Context analysis - check if we're in a pause region
        context_window = min(20, ctx.n_frames // 20)
        pause_score = self._detect_pause_region(ctx, context_window)
        
        # Weighted combination (adjusted for vocal importance)
        score = (
            energy_score * 0.20 +
            flux_score * 0.18 +
            onset_score * 0.18 +
            mfcc_score * 0.15 +
            centroid_score * 0.12 +
            melodic_score * 0.10 +
            pause_score * 0.07
        )
        
        # Bonus if both parts have vocals and are well-matched
        if has_vocal_start and has_vocal_end and mfcc_score > 0.8:
            score = min(1.0, score + 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _detect_vocal_presence(self, ctx: TransitionContext, is_start: bool) -> bool:
        """
        Detect if vocals are present in the audio segment.
        
        Uses multiple indicators:
        - Spectral centroid in vocal range (200-2000 Hz)
        - MFCC characteristics (vocal formants)
        - Spectral rolloff (vocals have specific frequency distribution)
        - Harmonic content (vocals are harmonic)
        """
        try:
            # Get features for the appropriate part
            if is_start:
                centroid = ctx.centroid_start
                rolloff = ctx.rolloff_start
                mfcc = ctx.mfcc_start
                flatness = ctx.flatness_start
            else:
                centroid = ctx.centroid_end
                rolloff = ctx.rolloff_end
                mfcc = ctx.mfcc_end
                flatness = ctx.flatness_end
            
            # Vocal indicators
            indicators = []
            
            # 1. Centroid in vocal range (normalized: 0.05-0.5 for 200-2000 Hz at 44.1kHz)
            if 0.05 < centroid < 0.5:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 2. Rolloff in vocal range (vocals have energy up to ~4kHz)
            if 0.1 < rolloff < 0.6:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 3. Low spectral flatness (vocals are tonal, not noisy)
            if flatness < 0.3:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 4. MFCC characteristics (first few coefficients capture vocal formants)
            # Vocals typically have higher energy in first few MFCCs
            mfcc_energy = np.sum(np.abs(mfcc[:5])) / (np.sum(np.abs(mfcc)) + 1e-10)
            if mfcc_energy > 0.4:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # Context analysis - check surrounding frames
            context_vocal_score = self._check_context_vocals(ctx, is_start)
            indicators.append(context_vocal_score)
            
            # If majority of indicators suggest vocals, return True
            vocal_score = np.mean(indicators)
            return vocal_score > 0.5
            
        except Exception:
            return False
    
    def _check_context_vocals(self, ctx: TransitionContext, is_start: bool) -> float:
        """Check vocal presence in context window."""
        try:
            sr = ctx.sr
            hop = ctx.hop
            
            if is_start:
                frame = ctx.start_frame
            else:
                frame = ctx.end_frame
            
            # Get context window (larger for better detection)
            context_frames = min(30, ctx.n_frames // 10)
            start_frame = max(0, frame - context_frames)
            end_frame = min(ctx.n_frames, frame + context_frames)
            
            # Get audio segment
            start_sample = int(start_frame * hop)
            end_sample = int(end_frame * hop)
            
            if end_sample <= start_sample or end_sample > len(ctx.audio):
                return 0.5
            
            segment = ctx.audio[start_sample:end_sample]
            
            if len(segment) < 64:
                return 0.5
            
            # Analyze spectral characteristics
            # Use FFT to check for vocal formants
            n_fft = 2 ** int(np.ceil(np.log2(len(segment))))
            fft = np.abs(rfft(segment * np.hanning(len(segment)), n=n_fft))
            freqs = np.fft.fftfreq(n_fft, 1/sr)[:len(fft)]
            
            # Check energy in vocal frequency range (200-2000 Hz)
            vocal_mask = (freqs >= 200) & (freqs <= 2000)
            if np.any(vocal_mask):
                vocal_energy = np.sum(fft[vocal_mask])
                total_energy = np.sum(fft)
                vocal_ratio = vocal_energy / (total_energy + 1e-10)
                
                # High ratio = likely vocals
                return float(vocal_ratio)
            
            return 0.5
        except Exception:
            return 0.5
    
    def _evaluate_melodic_continuity(self, ctx: TransitionContext, has_vocal_start: bool, has_vocal_end: bool) -> float:
        """Evaluate melodic continuity for vocal lines."""
        if not (has_vocal_start and has_vocal_end):
            return 0.5  # Not applicable if no vocals
        
        # Check chroma similarity (melodic content)
        chroma_sim = np.dot(ctx.chroma_start, ctx.chroma_end) / (
            np.linalg.norm(ctx.chroma_start) * np.linalg.norm(ctx.chroma_end) + 1e-10
        )
        
        # Check if chroma patterns suggest melodic continuity
        # Vocals typically have strong chroma in specific keys
        return float(chroma_sim)
    
    def _detect_pause_region(self, ctx: TransitionContext, window: int) -> float:
        """Detect if transition is in a pause region between words/phrases."""
        try:
            # Get audio segments around transition points
            sr = ctx.sr
            hop = ctx.hop
            
            # Convert frames to samples
            start_sample = int(ctx.start_frame * hop)
            end_sample = int(ctx.end_frame * hop)
            
            # Get context windows
            window_samples = int(window * hop)
            start_ctx_start = max(0, start_sample - window_samples)
            start_ctx_end = min(len(ctx.audio), start_sample + window_samples)
            end_ctx_start = max(0, end_sample - window_samples)
            end_ctx_end = min(len(ctx.audio), end_sample + window_samples)
            
            # Extract audio segments
            start_seg = ctx.audio[start_ctx_start:start_ctx_end]
            end_seg = ctx.audio[end_ctx_start:end_ctx_end]
            
            if len(start_seg) < 32 or len(end_seg) < 32:
                return 0.5
            
            # Compute RMS energy in context
            start_rms = np.sqrt(np.mean(start_seg ** 2))
            end_rms = np.sqrt(np.mean(end_seg ** 2))
            
            # Low energy = pause region
            avg_rms = (start_rms + end_rms) / 2
            max_rms = max(np.max(np.abs(start_seg)), np.max(np.abs(end_seg)))
            
            # Normalize
            if max_rms > 0:
                pause_score = 1.0 - min(1.0, avg_rms / (max_rms + 1e-10))
            else:
                pause_score = 0.5
            
            return pause_score
        except Exception:
            return 0.5
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        avg_energy = (ctx.rms_start + ctx.rms_end) / 2
        avg_flux = (ctx.flux_start + ctx.flux_end) / 2
        return (
            f"Vocal: {score:.2f} | "
            f"Energy: {avg_energy:.3f} | "
            f"Flux: {avg_flux:.3f}"
        )


@njit(cache=True, fastmath=True)
def detect_vocal_pauses(
    rms: np.ndarray,
    flux: np.ndarray,
    onset: np.ndarray,
    window: int = 10
) -> np.ndarray:
    """
    Detect vocal pause regions using energy and spectral stability.
    
    Returns array of pause probabilities (0-1) for each frame.
    """
    n = len(rms)
    pause_scores = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        # Get local context
        start = max(0, i - window)
        end = min(n, i + window + 1)
        
        # Local energy
        local_rms = np.mean(rms[start:end])
        local_flux = np.mean(flux[start:end])
        local_onset = np.mean(onset[start:end])
        
        # Pause = low energy, low flux, low onset
        pause_score = (
            (1.0 - min(1.0, local_rms * 1.5)) * 0.4 +
            (1.0 - min(1.0, local_flux * 2.0)) * 0.35 +
            (1.0 - min(1.0, local_onset * 2.0)) * 0.25
        )
        
        pause_scores[i] = max(0.0, min(1.0, pause_score))
    
    return pause_scores

