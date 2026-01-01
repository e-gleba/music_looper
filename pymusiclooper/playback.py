"""Module for playback through the terminal"""
import importlib
import logging
import signal
import threading

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn

from pymusiclooper.console import rich_console

# Lazy-load sounddevice: call `sd()` to return the module.
# (We use this instead of `lazy-loader`, to get clearer error messages when
# sounddevice is missing dependencies like PortAudio.)
_sd = None
def sd():
    global _sd
    _sd = _sd or importlib.import_module("sounddevice")
    return _sd

class PlaybackHandler:
    """Handler class for initiating looping playback through the terminal."""
    def __init__(self) -> None:
        self.event = threading.Event()
        self.progressbar = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[time_field]}"),
            TextColumn("{task.fields[loop_field]}"),
            console=rich_console,
            transient=True,
            refresh_per_second=2,
        )

    def play_looping(
        self,
        playback_data: np.ndarray,
        samplerate: int,
        n_channels: int,
        loop_start: int,
        loop_end: int,
        start_from=0,
    ) -> None:
        """Plays an audio track through the terminal with a loop active between the `loop_start` and `loop_end` provided. Ctrl+C to interrupt.

        Args:
            playback_data (np.ndarray): A numpy array containing the playback audio. Must be in the shape (samples, channels).
            samplerate (int): The sample rate of the playback
            n_channels (int): The number of channels for playback
            loop_start (int): The start point of the loop (in samples)
            loop_end (int): The end point of the loop (in samples)
            start_from (int, optional): The offset to start from (in samples). Defaults to 0.
        """
        self.loop_counter = 0
        self.looping = True
        self.current_frame = start_from

        total_samples = playback_data.shape[0]

        if loop_start > loop_end:
            raise ValueError(
                "Loop parameters are in the wrong order. "
                f"Loop start: {loop_start}; loop end: {loop_end}."
            )

        is_loop_invalid = (
            loop_start < 0
            or loop_start >= total_samples
            or loop_end < 0
            or loop_end >= total_samples
            or loop_start >= loop_end
        )

        if is_loop_invalid:
            raise ValueError(
                "Loop parameters are out of bounds. "
                f"Loop start: {loop_start}; "
                f"loop end: {loop_end}; "
                f"total number of samples in audio: {total_samples}."
            )

        try:

            def callback(outdata, frames, time, status):
                chunksize = min(len(playback_data) - self.current_frame, frames)

                # Audio looping logic with smooth crossfade
                if self.looping and self.current_frame + frames > loop_end:
                    pre_loop_index = loop_end - self.current_frame
                    remaining_frames = frames - (loop_end - self.current_frame)
                    adjusted_next_frame_idx = loop_start + remaining_frames
                    
                    # Calculate crossfade length (adaptive but simple)
                    fade_ms = 30  # Default 30ms crossfade
                    fade_samples = int(samplerate * fade_ms / 1000)
                    fade_samples = min(fade_samples, pre_loop_index, remaining_frames)
                    
                    if fade_samples > 16 and pre_loop_index > fade_samples and remaining_frames > fade_samples:
                        # Apply crossfade
                        pre_segment = playback_data[self.current_frame : loop_end]
                        post_segment = playback_data[loop_start:adjusted_next_frame_idx]
                        
                        try:
                            # Use adaptive transition selector for playback (best quality with effects)
                            # ALWAYS detects and masks rhythm gaps
                            try:
                                from pymusiclooper.transitions_effects import adaptive_transition_selector
                                crossfaded = adaptive_transition_selector(
                                    pre_segment[-fade_samples:],
                                    post_segment[:fade_samples],
                                    samplerate,
                                    fade_samples,
                                    rhythm_score=0.7,  # Can be improved with context
                                    harmonic_score=0.7,
                                    transient_strength=0.4,
                                    has_vocals=False,
                                )
                            except Exception:
                                # Fallback to perceptual crossfade
                                try:
                                    from pymusiclooper.transitions_advanced import perceptual_crossfade
                                    crossfaded = perceptual_crossfade(
                                        pre_segment[-fade_samples:],
                                        post_segment[:fade_samples],
                                        samplerate,
                                        fade_samples
                                    )
                                except Exception:
                                    # Final fallback to cosine crossfade
                                    from pymusiclooper.transitions import cosine_crossfade
                                    crossfaded = cosine_crossfade(
                                        pre_segment[-fade_samples:],
                                        post_segment[:fade_samples],
                                        fade_samples
                                    )
                            
                            # Copy audio with crossfade
                            if pre_loop_index > fade_samples:
                                outdata[:pre_loop_index - fade_samples] = (
                                    playback_data[self.current_frame : loop_end - fade_samples]
                                )
                            
                            # Crossfaded section
                            crossfade_start = pre_loop_index - fade_samples
                            crossfade_end = crossfade_start + len(crossfaded)
                            if crossfade_end <= frames:
                                outdata[crossfade_start:crossfade_end] = crossfaded
                                
                                # Rest of post segment
                                if remaining_frames > fade_samples:
                                    post_start = fade_samples
                                    post_end = min(remaining_frames, frames - crossfade_end)
                                    if post_end > post_start:
                                        outdata[crossfade_end:crossfade_end + (post_end - post_start)] = (
                                            post_segment[post_start:post_start + (post_end - post_start)]
                                        )
                            else:
                                # Crossfade doesn't fit, use simple copy
                                outdata[:pre_loop_index] = pre_segment
                                outdata[pre_loop_index:frames] = post_segment[:remaining_frames]
                        except Exception:
                            # Fallback to simple copy if crossfade fails
                            outdata[:pre_loop_index] = playback_data[self.current_frame : loop_end]
                            outdata[pre_loop_index:frames] = playback_data[loop_start:adjusted_next_frame_idx]
                    else:
                        # Too short for crossfade, use simple copy
                        outdata[:pre_loop_index] = playback_data[self.current_frame : loop_end]
                        outdata[pre_loop_index:frames] = playback_data[loop_start:adjusted_next_frame_idx]
                    
                    self.current_frame = adjusted_next_frame_idx
                    self.loop_counter += 1
                    rich_console.print(f"[dim italic yellow]Currently on loop #{self.loop_counter}.[/]", end="\r")
                else:
                    outdata[:chunksize] = playback_data[self.current_frame : self.current_frame + chunksize]
                    self.current_frame += chunksize
                    if chunksize < frames:
                        outdata[chunksize:] = 0
                        raise sd().CallbackStop()

            self.stream = sd().OutputStream(
                samplerate=samplerate,
                channels=n_channels,
                callback=callback,
                finished_callback=self.event.set,
            )

            with self.stream, self.progressbar:
                # Initialize playback progress bar
                pbar = self.progressbar.add_task(
                    "Now Playing...",
                    total=total_samples,
                    loop_field="",
                    time_field="",
                )

                # Override SIGINT/KeyboardInterrupt handler with custom logic for loop handling
                signal.signal(signal.SIGINT, self._loop_interrupt_handler)

                # Workaround for python issue on Windows
                # (threading.Event().wait() not interruptable with Ctrl-C on Windows): https://bugs.python.org/issue35935
                # Set a 0.5 second timeout to handle interrupts in-between
                while not self.event.wait(0.5):
                    # Update playback progress bar between wait timeouts
                    time_sec = self.current_frame / samplerate
                    ftime = f"{time_sec // 60:02.0f}:{time_sec % 60:02.0f}"
                    self.progressbar.update(
                        pbar,
                        completed=self.current_frame,
                        time_field=ftime,
                        loop_field=(
                            f"[dim] | Currently on Loop #{self.loop_counter}[/]"
                            if self.loop_counter
                            else ""
                        ),
                    )

                # Restore default SIGINT handler after playback is stopped
                signal.signal(signal.SIGINT, signal.default_int_handler)
        except Exception as e:
            logging.error(e)

    def _loop_interrupt_handler(self, *args):
        if self.looping:
            self.looping = False
            rich_console.print("[dim italic yellow](Looping disabled. [red]Ctrl+C[/] again to stop playback.)[/]")
        else:
            self.event.set()
            self.stream.stop()
            self.stream.close()
            rich_console.print("[dim]Playback interrupted by user.[/]")
            # Restore default SIGINT handler
            signal.signal(signal.SIGINT, signal.default_int_handler)

    def play_multi_hop(
        self,
        playback_data: np.ndarray,
        samplerate: int,
        n_channels: int,
        segments: list,
    ) -> None:
        """Plays multiple segments in sequence, then loops back to the first.
        
        Args:
            playback_data: Audio data array (samples, channels)
            samplerate: Sample rate
            n_channels: Number of channels
            segments: List of (start_sample, end_sample) tuples
        """
        if not segments:
            return
            
        self.loop_counter = 0
        self.looping = True
        self.current_segment_idx = 0
        self.segments = segments
        
        # Start from first segment
        first_start, first_end = segments[0]
        self.current_frame = first_start
        self.segment_end = first_end
        
        total_samples = playback_data.shape[0]

        try:
            def callback(outdata, frames, time, status):
                remaining = frames
                out_offset = 0
                
                while remaining > 0:
                    # How much left in current segment
                    seg_remaining = self.segment_end - self.current_frame
                    
                    if seg_remaining <= 0:
                        # Move to next segment
                        if self.looping:
                            self.current_segment_idx = (self.current_segment_idx + 1) % len(self.segments)
                            if self.current_segment_idx == 0:
                                self.loop_counter += 1
                                rich_console.print(f"[dim italic yellow]Loop #{self.loop_counter} - playing segment chain...[/]", end="\r")
                        else:
                            # Not looping, stop
                            outdata[out_offset:] = 0
                            raise sd().CallbackStop()
                        
                        seg_start, seg_end = self.segments[self.current_segment_idx]
                        self.current_frame = seg_start
                        self.segment_end = seg_end
                        seg_remaining = seg_end - seg_start
                    
                    # Copy audio with crossfade at segment boundaries
                    to_copy = min(remaining, seg_remaining)
                    
                    # Check if we're at a segment boundary (need crossfade)
                    if seg_remaining <= to_copy and self.looping:
                        # We're at the end of current segment, apply crossfade
                        next_seg_idx = (self.current_segment_idx + 1) % len(self.segments)
                        next_start, next_end = self.segments[next_seg_idx]
                        
                        from pymusiclooper.transitions import cosine_crossfade
                        
                        # Calculate crossfade length
                        fade_ms = 25  # 25ms crossfade for multi-hop
                        fade_samples = int(samplerate * fade_ms / 1000)
                        fade_samples = min(fade_samples, to_copy // 2)
                        
                        if fade_samples > 16 and to_copy > fade_samples * 2:
                            # Apply crossfade
                            current_seg = playback_data[self.current_frame:self.current_frame + to_copy]
                            next_seg = playback_data[next_start:min(len(playback_data), next_start + to_copy)]
                            
                            try:
                                # Crossfade only the overlapping part
                                crossfaded = cosine_crossfade(
                                    current_seg[-fade_samples:],
                                    next_seg[:fade_samples],
                                    fade_samples
                                )
                                
                                # Copy non-crossfaded parts
                                if to_copy > fade_samples:
                                    outdata[out_offset:out_offset + (to_copy - fade_samples)] = (
                                        current_seg[:-(fade_samples)]
                                    )
                                
                                # Copy crossfaded part
                                crossfade_start = out_offset + (to_copy - fade_samples)
                                crossfade_end = crossfade_start + len(crossfaded)
                                if crossfade_end <= len(outdata):
                                    outdata[crossfade_start:crossfade_end] = crossfaded
                                
                                self.current_frame += to_copy
                                out_offset += min(to_copy, remaining)
                                remaining -= min(to_copy, remaining)
                            except Exception:
                                # Fallback to simple copy
                                outdata[out_offset:out_offset + to_copy] = (
                                    playback_data[self.current_frame:self.current_frame + to_copy]
                                )
                                self.current_frame += to_copy
                                out_offset += to_copy
                                remaining -= to_copy
                        else:
                            # Too short for crossfade, use simple copy
                            outdata[out_offset:out_offset + to_copy] = (
                                playback_data[self.current_frame:self.current_frame + to_copy]
                            )
                            self.current_frame += to_copy
                            out_offset += to_copy
                            remaining -= to_copy
                    else:
                        # Normal copy
                        outdata[out_offset:out_offset + to_copy] = playback_data[self.current_frame:self.current_frame + to_copy]
                        self.current_frame += to_copy
                        out_offset += to_copy
                        remaining -= to_copy

            self.stream = sd().OutputStream(
                samplerate=samplerate,
                channels=n_channels,
                callback=callback,
                finished_callback=self.event.set,
            )

            with self.stream, self.progressbar:
                # Initialize playback progress bar
                total_chain_samples = sum(e - s for s, e in segments)
                pbar = self.progressbar.add_task(
                    f"Multi-hop ({len(segments)} segments)...",
                    total=total_chain_samples,
                    loop_field="",
                    time_field="",
                )

                signal.signal(signal.SIGINT, self._loop_interrupt_handler)

                while not self.event.wait(0.5):
                    # Calculate progress within chain
                    progress = 0
                    for i, (s, e) in enumerate(self.segments):
                        if i < self.current_segment_idx:
                            progress += e - s
                        elif i == self.current_segment_idx:
                            progress += self.current_frame - s
                    
                    time_sec = progress / samplerate
                    ftime = f"{time_sec // 60:02.0f}:{time_sec % 60:02.0f}"
                    seg_info = f"Seg {self.current_segment_idx + 1}/{len(segments)}"
                    self.progressbar.update(
                        pbar,
                        completed=progress,
                        time_field=ftime,
                        loop_field=(
                            f"[dim] | {seg_info} | Loop #{self.loop_counter}[/]"
                            if self.loop_counter
                            else f"[dim] | {seg_info}[/]"
                        ),
                    )

                signal.signal(signal.SIGINT, signal.default_int_handler)
        except Exception as e:
            logging.error(e)
