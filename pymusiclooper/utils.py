"""General utility functions."""

from pathlib import Path

from pymusiclooper.youtube import YoutubeDownloader

DEFAULT_OUTPUT_DIR = "LooperOutput"


def get_outputdir(path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Returns the absolute output directory path for exports.

    Args:
        path: The file or directory being processed.
        output_dir: Custom output directory. If None, uses 'LooperOutput' in path's directory.

    Returns:
        Absolute path to the output directory.
    """
    if output_dir is not None:
        return Path(output_dir).resolve()

    p = Path(path)
    base = p if p.is_dir() else p.parent
    return (base / DEFAULT_OUTPUT_DIR).resolve()


def mk_outputdir(path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Creates and returns the output directory.

    Args:
        path: The file or directory being processed.
        output_dir: Custom output directory. If None, creates 'LooperOutput' in path's directory.

    Returns:
        Absolute path to the created output directory.
    """
    out = get_outputdir(path, output_dir)
    out.mkdir(exist_ok=True)
    return out


def download_audio(url: str, output_dir: str | Path) -> str:
    """Downloads audio from URL using yt-dlp.

    Args:
        url: The URL to extract audio from.
        output_dir: Directory to store the downloaded audio.

    Returns:
        Filepath of the extracted audio.
    """
    return YoutubeDownloader(url, str(output_dir)).filepath
