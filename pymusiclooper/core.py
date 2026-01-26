import os
import shutil
from pathlib import Path

import lazy_loader as lazy
import numpy as np

from pymusiclooper.analysis import LoopPair, find_best_loop_points
from pymusiclooper.audio import MLAudio
from pymusiclooper.playback import PlaybackHandler

soundfile = lazy.load("soundfile")


class MusicLooper:
    """High-level API access to PyMusicLooper's main functions."""

    def __init__(self, filepath: str):
        """Initializes the MusicLooper object with the provided audio track.

        Args:
            filepath (str): path to the audio track to use.
        """
        self.mlaudio = MLAudio(filepath=filepath)

    def find_loop_pairs(
        self,
        min_duration_multiplier: float = 0.35,
        min_loop_duration: float | None = None,
        max_loop_duration: float | None = None,
        approx_loop_start: float | None = None,
        approx_loop_end: float | None = None,
        brute_force: bool = False,
        disable_pruning: bool = False,
    ) -> list[LoopPair]:
        return find_best_loop_points(
            mlaudio=self.mlaudio,
            min_duration_multiplier=min_duration_multiplier,
            min_loop_duration=min_loop_duration,
            max_loop_duration=max_loop_duration,
            approx_loop_start=approx_loop_start,
            approx_loop_end=approx_loop_end,
            brute_force=brute_force,
            disable_pruning=disable_pruning,
        )

    @property
    def filename(self) -> str:
        return self.mlaudio.filename

    @property
    def filepath(self) -> str:
        return self.mlaudio.filepath

    def samples_to_frames(self, samples: int) -> int:
        return self.mlaudio.samples_to_frames(samples)

    def samples_to_seconds(self, samples: int) -> float:
        return self.mlaudio.samples_to_seconds(samples)

    def frames_to_samples(self, frame: int) -> int:
        return self.mlaudio.frames_to_samples(frame)

    def seconds_to_frames(self, seconds: float) -> int:
        return self.mlaudio.seconds_to_frames(seconds)

    def seconds_to_samples(self, seconds: float) -> int:
        return self.mlaudio.seconds_to_samples(seconds)

    def frames_to_ftime(self, frame: int) -> str:
        return self.mlaudio.frames_to_ftime(frame)

    def samples_to_ftime(self, samples: int) -> str:
        return self.mlaudio.samples_to_ftime(samples)

    def play_looping(self, loop_start: int, loop_end: int, start_from: int = 0):
        playback_handler = PlaybackHandler()
        playback_handler.play_looping(
            self.mlaudio.playback_audio,
            self.mlaudio.rate,
            self.mlaudio.n_channels,
            loop_start,
            loop_end,
            start_from,
        )

    def export(
        self,
        loop_start: int,
        loop_end: int,
        format: str = "WAV",
        output_dir: str | None = None,
    ):
        if output_dir is not None:
            out_path = os.path.join(output_dir, self.mlaudio.filename)
        else:
            out_path = os.path.abspath(self.mlaudio.filepath)

        soundfile.write(
            f"{out_path}-intro.{format.lower()}",
            self.mlaudio.playback_audio[:loop_start],
            self.mlaudio.rate,
            format=format,
        )
        soundfile.write(
            f"{out_path}-loop.{format.lower()}",
            self.mlaudio.playback_audio[loop_start:loop_end],
            self.mlaudio.rate,
            format=format,
        )
        soundfile.write(
            f"{out_path}-outro.{format.lower()}",
            self.mlaudio.playback_audio[loop_end:],
            self.mlaudio.rate,
            format=format,
        )

    def extend(
        self,
        loop_start: int,
        loop_end: int,
        extended_length: float,
        fade_length: float = 5.0,
        disable_fade_out: bool = False,
        format: str = "WAV",
        output_dir: str | Path | None = None,
    ) -> str:
        audio = self.mlaudio.playback_audio

        if extended_length < self.mlaudio.total_duration:
            raise ValueError("Extended length must exceed original audio duration.")

        out_path = (
            Path(output_dir) / self.mlaudio.filename
            if output_dir
            else Path(self.mlaudio.filepath).resolve()
        )

        # Split audio sections
        intro = audio[:loop_start]
        loop = audio[loop_start:loop_end]
        outro = audio[loop_end:]

        # Calculate loop repetitions needed
        target_samples = self.mlaudio.seconds_to_samples(extended_length) - len(intro)
        if disable_fade_out:
            target_samples -= len(outro)

        n_full_loops = int(target_samples // len(loop))
        leftover_ratio = (target_samples / len(loop)) - n_full_loops
        leftover_end = loop_start + int((loop_end - loop_start) * leftover_ratio)

        # Build final loop section
        if disable_fade_out:
            final_section = loop
        else:
            final_section = audio[loop_start:leftover_end].copy()
            fade_samples = min(
                self.mlaudio.seconds_to_samples(fade_length), len(final_section)
            )
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)[:, np.newaxis]
                final_section[-fade_samples:] *= fade

        # Format output filename with duration
        total_samples = len(intro) + (n_full_loops * len(loop)) + len(final_section)
        if disable_fade_out:
            total_samples += len(outro)

        total_secs = int(self.mlaudio.samples_to_seconds(total_samples))
        duration_str = f"{total_secs // 60}m{total_secs % 60:02d}s"
        output_file = out_path.with_stem(
            f"{out_path.stem}-extended-{duration_str}"
        ).with_suffix(f".{format.lower()}")

        # Buffered write (avoids loading full extended audio into memory)
        dtype = str(audio.dtype)
        with soundfile.SoundFile(
            str(output_file),
            mode="w",
            samplerate=self.mlaudio.rate,
            channels=self.mlaudio.n_channels,
            format=format,
        ) as sf:
            sf.buffer_write(intro.tobytes(order="C"), dtype)
            for _ in range(n_full_loops):
                sf.buffer_write(loop.tobytes(order="C"), dtype)
            sf.buffer_write(final_section.tobytes(order="C"), dtype)
            if disable_fade_out:
                sf.buffer_write(outro.tobytes(order="C"), dtype)

        # Copy metadata tags (best-effort)
        self._copy_tags(output_file)

        return str(output_file)

    def _copy_tags(self, dest_path: Path) -> None:
        """Attempt to copy audio tags from source to destination."""
        try:
            import taglib

            with taglib.File(self.mlaudio.filepath, save_on_exit=False) as src:
                tags = src.tags
            with taglib.File(str(dest_path), save_on_exit=True) as dst:
                dst.tags.update(tags)
        except Exception:
            pass  # Silently ignore; tags are optional

    def export_txt(
        self,
        loop_start: int | float | str,
        loop_end: str | int | float | str,
        txt_name: str = "loops",
        output_dir: str | None = None,
    ):
        if output_dir is not None:
            out_path = os.path.join(output_dir, f"{txt_name}.txt")
        else:
            out_path = os.path.join(
                os.path.dirname(self.mlaudio.filepath), f"{txt_name}.txt"
            )

        with open(out_path, "a") as file:
            file.write(f"{loop_start} {loop_end} {self.mlaudio.filename}\n")

    def _find_start_tag(
        self,
        file_tags: dict[str, list[str]],
    ) -> str:
        # List derived from:
        # https://github.com/libsdl-org/SDL_mixer/blob/5175907b515ea9e07d0b35849bfaf09870d07d33/src/codecs/music_ogg.c#L289-L302
        # https://github.com/vgmstream/vgmstream/blob/02d3c3f875fb97b682c4479fe66c7e0a0eeee04d/src/meta/ogg_vorbis.c#L647-L675
        known_tags = [
            "COMMENT=LOOPPOINT",
            "LOOP",
            "LOOP_BEGIN",
            "LOOPPOINT",
            "LOOPS",
            "LOOPSTART",
            "LOOP_START",
            "LOOP-START",
            "UM3.STREAM.LOOPPOINT.START",
            "XIPH_CUE_LOOPSTART",
        ]

        for tag in known_tags:
            if tag in file_tags:
                return tag

        raise ValueError(
            f'No loop start tag could be automatically detected in the metadata of "{self.filename}".'
        )

    def _find_end_tag(
        self,
        file_tags: dict[str, list[str]],
    ) -> str:
        known_tags = [
            "LOOPE",
            "LOOPEND",
            "LOOP_END",
            "LOOP-END",
            "LOOPLENGTH",
            "LOOP_LENGTH",
            "LOOP-LENGTH",
            "XIPH_CUE_LOOPEND",
        ]

        for tag in known_tags:
            if tag in file_tags:
                return tag

        raise ValueError(
            f'No loop end tag could be automatically detected in the metadata of "{self.filename}".'
        )

    def _end_tag_is_offset(
        self,
        loop_end_tag: str,
        is_offset: bool | None,
    ) -> bool:
        if is_offset is not None:
            return is_offset

        upper_loop_end_tag = loop_end_tag.upper()

        return "LEN" in upper_loop_end_tag or "OFFSET" in upper_loop_end_tag

    def export_tags(
        self,
        loop_start: int,
        loop_end: int,
        loop_start_tag: str,
        loop_end_tag: str,
        is_offset: bool | None = None,
        output_dir: str | None = None,
    ) -> tuple[str]:
        # Workaround for taglib import issues on Apple silicon devices
        # Import taglib only when needed to isolate ImportErrors
        import taglib

        if output_dir is None:
            output_dir = os.path.abspath(self.mlaudio.filepath)

        track_name, file_extension = os.path.splitext(self.mlaudio.filename)

        exported_file_path = os.path.join(
            output_dir, f"{track_name}-tagged{file_extension}"
        )
        shutil.copyfile(self.mlaudio.filepath, exported_file_path)

        # Handle LOOPLENGTH tag
        if self._end_tag_is_offset(loop_end_tag, is_offset):
            loop_end = loop_end - loop_start

        with taglib.File(exported_file_path, save_on_exit=True) as audio_file:
            audio_file.tags[loop_start_tag] = [str(loop_start)]
            audio_file.tags[loop_end_tag] = [str(loop_end)]

        return str(loop_start), str(loop_end)

    def read_tags(
        self, loop_start_tag: str, loop_end_tag: str, is_offset: bool | None = None
    ) -> tuple[int, int]:
        """Reads the tags provided from the file and returns the read loop points

        Args:
            loop_start_tag (str): The name of the metadata tag containing the loop_start value
            loop_end_tag (str): The name of the metadata tag containing the loop_end value
            is_offset (bool, optional): Parse second tag as relative length / absolute end. Defaults to auto-detecting based on tag name.

        Returns:
            Tuple[int, int]: A tuple containing (loop_start, loop_end)
        """
        # Workaround for taglib import issues on Apple silicon devices
        # Import taglib only when needed to isolate ImportErrors
        import taglib

        loop_start = None
        loop_end = None

        with taglib.File(self.filepath) as audio_file:
            if loop_start_tag is None:
                loop_start_tag = self._find_start_tag(audio_file.tags)
            if loop_end_tag is None:
                loop_end_tag = self._find_end_tag(audio_file.tags)
            if loop_start_tag not in audio_file.tags:
                raise ValueError(
                    f'The tag "{loop_start_tag}" is not present in the metadata of "{self.filename}".'
                )
            if loop_end_tag not in audio_file.tags:
                raise ValueError(
                    f'The tag "{loop_end_tag}" is not present in the metadata of "{self.filename}".'
                )
            try:
                loop_start = int(audio_file.tags[loop_start_tag][0])
                loop_end = int(audio_file.tags[loop_end_tag][0])
            except Exception as e:
                raise TypeError(
                    "One of the tags provided has invalid (non-integer or empty) values"
                ) from e

        # Re-order the loop points in case
        real_loop_start = min(loop_start, loop_end)
        real_loop_end = max(loop_start, loop_end)

        # Handle LOOPLENGTH tag
        if self._end_tag_is_offset(loop_end_tag, is_offset):
            real_loop_end = real_loop_start + real_loop_end

        return real_loop_start, real_loop_end
