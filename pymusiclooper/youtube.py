import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _yt_dlp():
    """Lazy-load yt_dlp module."""
    import yt_dlp

    return yt_dlp


class YtdLogger:
    """Minimal logger for yt-dlp with optional verbose mode."""

    __slots__ = ("verbose",)

    def __init__(self) -> None:
        self.verbose = os.getenv("PML_VERBOSE") is not None

    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug] "):
            self.info(msg)

    def info(self, msg: str) -> None:
        match msg:
            case s if "(pass -k to keep)" in s:
                pass
            case s if s.startswith("[download]"):
                print(s, end="\r")
            case s if s.startswith("[ExtractAudio]"):
                print(s)
            case _ if self.verbose:
                print(msg)

    def warning(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def error(self, msg: str) -> None:
        print(msg)


class YoutubeDownloader:
    """Downloads and extracts audio from YouTube URLs using yt-dlp."""

    __slots__ = ("filepath", "error_code")

    # SponsorBlock segments to skip (non-music content)
    SKIP_SEGMENTS = (
        "sponsor",
        "selfpromo",
        "interaction",
        "intro",
        "outro",
        "preview",
        "music_offtopic",
        "filler",
    )

    def __init__(self, url: str, output_path: str) -> None:
        self.filepath: str | None = None

        opts = {
            "logger": YtdLogger(),
            "format": "bestaudio/best",
            "paths": {"home": output_path, "temp": output_path},
            "progress_hooks": [self._on_progress],
            "postprocessor_hooks": [self._on_postprocess],
            "postprocessors": [
                {"key": "SponsorBlock", "when": "pre_process"},
                {
                    "key": "ModifyChapters",
                    "remove_sponsor_segments": list(self.SKIP_SEGMENTS),
                },
                {"key": "FFmpegExtractAudio"},
            ],
        }

        with _yt_dlp().YoutubeDL(opts) as ydl:
            self.error_code = ydl.download([url])

    def _on_progress(self, d: dict) -> None:
        if d["status"] == "finished":
            print("\nDone downloading, now post-processing...")

    def _on_postprocess(self, d: dict) -> None:
        if d["status"] == "finished":
            self.filepath = d["info_dict"].get("filepath")
