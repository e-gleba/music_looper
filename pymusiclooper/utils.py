from pathlib import Path

from pymusiclooper.youtube import YoutubeDownloader

DEFAULT_OUTPUT_DIR = "LooperOutput"


def get_outputdir(path: str | Path, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()

    p = Path(path)
    base = p if p.is_dir() else p.parent
    return (base / DEFAULT_OUTPUT_DIR).resolve()


def mk_outputdir(path: str | Path, output_dir: str | Path | None = None) -> Path:
    out = get_outputdir(path, output_dir)
    out.mkdir(exist_ok=True)
    return out


def download_audio(url: str, output_dir: str | Path) -> str:
    return YoutubeDownloader(url, str(output_dir)).filepath
