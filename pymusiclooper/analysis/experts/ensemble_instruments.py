"""
Ensemble Instruments Expert - Detects and Ensures Continuity of Ensemble Instruments.

Prevents transitions where ensemble instruments (brass, woodwinds, strings, etc.)
appear or disappear, which creates audible breaks.
"""

from __future__ import annotations

import numpy as np

from pymusiclooper.analysis.experts.base import Expert, TransitionContext


class EnsembleInstrumentsExpert(Expert):
    """Expert for detecting ensemble instrument continuity.
    
    CRITICAL: Prevents transitions where ensemble instruments appear/disappear.
    """
    
    name = "ensemble_instruments"
    weight = 0.18  # High weight - critical for ensemble music
    
    def score(self, ctx: TransitionContext) -> float:
        """
        Evaluate transition quality for ensemble instruments.
        
        CRITICAL: Rejects transitions where ensemble instruments appear in one part but not the other.
        This prevents audible breaks when instruments start/stop.
        """
        # Detect ensemble instruments in both parts
        ensemble_start = self._detect_ensemble_instruments(ctx, is_start=True)
        ensemble_end = self._detect_ensemble_instruments(ctx, is_start=False)
        
        # HARD REJECTION: If ensemble instruments in one part but not the other
        if ensemble_start != ensemble_end:
            # One part has ensemble, other doesn't - this creates audible break
            return 0.05  # Almost reject - very bad transition
        
        # If both have ensemble or both don't, continue with normal scoring
        # Check instrument type consistency
        instrument_match = self._check_instrument_consistency(ctx, ensemble_start, ensemble_end)
        
        # Check spectral continuity
        spectral_continuity = self._check_spectral_continuity(ctx)
        
        # Check energy flow (ensemble instruments have specific energy patterns)
        energy_flow = self._check_energy_flow(ctx)
        
        # Combined score
        if ensemble_start and ensemble_end:
            # Both have ensemble - need good match
            score = (
                instrument_match * 0.40 +
                spectral_continuity * 0.35 +
                energy_flow * 0.25
            )
        else:
            # No ensemble - less critical, but still check consistency
            score = (
                instrument_match * 0.30 +
                spectral_continuity * 0.40 +
                energy_flow * 0.30
            )
        
        return max(0.0, min(1.0, score))
    
    def _detect_ensemble_instruments(self, ctx: TransitionContext, is_start: bool) -> bool:
        """
        Detect if ensemble instruments (brass, woodwinds, strings) are present.
        
        Uses spectral characteristics:
        - Brass: Strong harmonics, energy in 200-2000 Hz, bright timbre
        - Woodwinds: Formant-like structure, energy in 300-3000 Hz
        - Strings: Rich harmonics, energy in 100-4000 Hz, smooth timbre
        """
        try:
            if is_start:
                centroid = ctx.centroid_start
                rolloff = ctx.rolloff_start
                bandwidth = ctx.bandwidth_start
                flatness = ctx.flatness_start
                mfcc = ctx.mfcc_start
            else:
                centroid = ctx.centroid_end
                rolloff = ctx.rolloff_end
                bandwidth = ctx.bandwidth_end
                flatness = ctx.flatness_end
                mfcc = ctx.mfcc_end
            
            indicators = []
            
            # 1. Centroid in ensemble range (200-3000 Hz, normalized)
            if 0.05 < centroid < 0.7:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 2. Rolloff in ensemble range
            if 0.15 < rolloff < 0.8:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 3. Moderate bandwidth (ensemble instruments have spread)
            if 0.1 < bandwidth < 0.6:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 4. Low flatness (ensemble instruments are tonal)
            if flatness < 0.4:
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 5. MFCC characteristics (ensemble instruments have rich timbre)
            # Check if MFCCs suggest rich harmonic content
            mfcc_variance = np.var(mfcc)
            if mfcc_variance > 0.1:  # Rich timbre
                indicators.append(1.0)
            else:
                indicators.append(0.0)
            
            # 6. Context analysis
            context_score = self._check_context_ensemble(ctx, is_start)
            indicators.append(context_score)
            
            # If majority suggest ensemble, return True
            ensemble_score = np.mean(indicators)
            return ensemble_score > 0.5
            
        except Exception:
            return False
    
    def _check_context_ensemble(self, ctx: TransitionContext, is_start: bool) -> float:
        """Check ensemble presence in context window."""
        try:
            sr = ctx.sr
            hop = ctx.hop
            
            if is_start:
                frame = ctx.start_frame
            else:
                frame = ctx.end_frame
            
            context_frames = min(30, ctx.n_frames // 10)
            start_frame = max(0, frame - context_frames)
            end_frame = min(ctx.n_frames, frame + context_frames)
            
            start_sample = int(start_frame * hop)
            end_sample = int(end_frame * hop)
            
            if end_sample <= start_sample or end_sample > len(ctx.audio):
                return 0.5
            
            segment = ctx.audio[start_sample:end_sample]
            
            if len(segment) < 64:
                return 0.5
            
            # Analyze spectral characteristics
            from scipy.fft import rfft
            n_fft = 2 ** int(np.ceil(np.log2(len(segment))))
            fft = np.abs(rfft(segment * np.hanning(len(segment)), n=n_fft))
            freqs = np.fft.fftfreq(n_fft, 1/sr)[:len(fft)]
            
            # Check energy in ensemble frequency range (200-3000 Hz)
            ensemble_mask = (freqs >= 200) & (freqs <= 3000)
            if np.any(ensemble_mask):
                ensemble_energy = np.sum(fft[ensemble_mask])
                total_energy = np.sum(fft)
                ensemble_ratio = ensemble_energy / (total_energy + 1e-10)
                
                return float(ensemble_ratio)
            
            return 0.5
        except Exception:
            return 0.5
    
    def _check_instrument_consistency(self, ctx: TransitionContext, has_ensemble_start: bool, has_ensemble_end: bool) -> float:
        """Check if instrument types are consistent."""
        if not (has_ensemble_start and has_ensemble_end):
            return 0.5  # Not applicable
        
        # Check MFCC similarity (instrument timbre)
        mfcc_diff = np.linalg.norm(ctx.mfcc_start - ctx.mfcc_end)
        mfcc_sim = 1.0 - min(1.0, mfcc_diff / 4.0)  # Stricter for ensemble
        
        # Check spectral centroid similarity (instrument brightness)
        centroid_diff = abs(ctx.centroid_start - ctx.centroid_end)
        centroid_sim = 1.0 - min(1.0, centroid_diff * 3.0)
        
        # Check bandwidth similarity (instrument spread)
        bandwidth_diff = abs(ctx.bandwidth_start - ctx.bandwidth_end)
        bandwidth_sim = 1.0 - min(1.0, bandwidth_diff * 2.0)
        
        return (mfcc_sim * 0.5 + centroid_sim * 0.3 + bandwidth_sim * 0.2)
    
    def _check_spectral_continuity(self, ctx: TransitionContext) -> float:
        """Check spectral continuity across transition."""
        # Check spectral flux (low = smooth)
        avg_flux = (ctx.flux_start + ctx.flux_end) / 2
        flux_score = 1.0 - min(1.0, avg_flux * 2.0)
        
        # Check rolloff similarity
        rolloff_diff = abs(ctx.rolloff_start - ctx.rolloff_end)
        rolloff_score = 1.0 - min(1.0, rolloff_diff * 2.0)
        
        return (flux_score * 0.6 + rolloff_score * 0.4)
    
    def _check_energy_flow(self, ctx: TransitionContext) -> float:
        """Check energy flow continuity."""
        rms_diff = abs(ctx.rms_start - ctx.rms_end) / (max(ctx.rms_start, ctx.rms_end) + 1e-10)
        rms_score = 1.0 - min(1.0, rms_diff)
        
        loudness_diff = abs(ctx.loudness_start - ctx.loudness_end)
        loudness_score = 1.0 - min(1.0, loudness_diff)
        
        return (rms_score * 0.6 + loudness_score * 0.4)
    
    def explain(self, ctx: TransitionContext) -> str:
        score = self.score(ctx)
        ensemble_start = self._detect_ensemble_instruments(ctx, is_start=True)
        ensemble_end = self._detect_ensemble_instruments(ctx, is_start=False)
        match = "✓" if ensemble_start == ensemble_end else "✗"
        return (
            f"Ensemble: {score:.2f} | "
            f"Match: {match} | "
            f"Start: {'yes' if ensemble_start else 'no'}, "
            f"End: {'yes' if ensemble_end else 'no'}"
        )

