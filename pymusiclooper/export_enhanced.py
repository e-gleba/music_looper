"""
Enhanced Export Module - Export loops with crossfade applied.

Provides export functionality with smooth transitions:
- Export loop with crossfade applied
- Export intro + seamless loop + outro
- Multiple format support
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lazy_loader as lazy
import numpy as np

if TYPE_CHECKING:
    from pymusiclooper.audio import MLAudio

soundfile = lazy.load("soundfile")

from pymusiclooper.transitions import cosine_crossfade, analyze_transition_quality


def export_loop_with_crossfade(
    mlaudio: MLAudio,
    loop_start: int,
    loop_end: int,
    output_path: str,
    crossfade_ms: int | None = None,
    format: str = "WAV",
) -> None:
    """
    Export a seamless loop with crossfade applied.
    
    Args:
        mlaudio: Audio object
        loop_start: Loop start in samples
        loop_end: Loop end in samples
        output_path: Output file path
        crossfade_ms: Crossfade length in milliseconds (auto if None)
        format: Audio format (WAV, FLAC, etc.)
    """
    audio = mlaudio.playback_audio
    sr = mlaudio.rate
    
    # Analyze transition to determine optimal crossfade
    if crossfade_ms is None:
        quality_info = analyze_transition_quality(audio, loop_start, loop_end, sr)
        crossfade_ms = quality_info['fade_ms']
    
    crossfade_samples = int(sr * crossfade_ms / 1000)
    crossfade_samples = min(crossfade_samples, (loop_end - loop_start) // 4)
    
    # Extract loop segment
    loop_segment = audio[loop_start:loop_end]
    
    # Create seamless loop with crossfade
    # Crossfade from end to start
    end_part = loop_segment[-crossfade_samples:]
    start_part = loop_segment[:crossfade_samples]
    
    # Apply crossfade
    crossfaded = cosine_crossfade(end_part, start_part, crossfade_samples)
    
    # Build seamless loop: [start...end-crossfade] + [crossfaded] + [start+crossfade...end]
    seamless_loop = np.concatenate([
        loop_segment[:-crossfade_samples],  # Most of the loop
        crossfaded,  # Crossfaded transition
        loop_segment[crossfade_samples:]  # Rest of the loop
    ])
    
    # Ensure it loops perfectly
    # The crossfaded section should make end seamlessly connect to start
    final_loop = np.concatenate([
        seamless_loop,
        seamless_loop  # Two iterations to verify seamless
    ])
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        soundfile.write(
            str(output_path),
            final_loop,
            sr,
            format=format
        )
        logging.info(f"Exported seamless loop to {output_path}")
    except Exception as e:
        logging.error(f"Export failed: {e}")
        raise


def export_intro_loop_outro(
    mlaudio: MLAudio,
    loop_start: int,
    loop_end: int,
    output_dir: str | None = None,
    crossfade_ms: int = 30,
    format: str = "WAV",
) -> tuple[str, str, str]:
    """
    Export intro, seamless loop, and outro sections.
    
    Returns:
        Tuple of (intro_path, loop_path, outro_path)
    """
    audio = mlaudio.playback_audio
    sr = mlaudio.rate
    base_path = Path(mlaudio.filepath)
    
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = base_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = base_path.stem
    
    # Intro (everything before loop)
    intro = audio[:loop_start]
    intro_path = output_dir / f"{stem}_intro.{format.lower()}"
    soundfile.write(str(intro_path), intro, sr, format=format)
    
    # Seamless loop with crossfade
    loop_path = output_dir / f"{stem}_loop.{format.lower()}"
    export_loop_with_crossfade(
        mlaudio, loop_start, loop_end, str(loop_path),
        crossfade_ms=crossfade_ms, format=format
    )
    
    # Outro (everything after loop)
    outro = audio[loop_end:]
    outro_path = output_dir / f"{stem}_outro.{format.lower()}"
    if len(outro) > 0:
        soundfile.write(str(outro_path), outro, sr, format=format)
    else:
        # Create empty outro
        empty = np.zeros((int(sr * 0.1), audio.shape[1] if len(audio.shape) > 1 else 1))
        soundfile.write(str(outro_path), empty, sr, format=format)
    
    return str(intro_path), str(loop_path), str(outro_path)


