import logging
import os
import sys
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.box import ROUNDED

from pymusiclooper.analysis import LoopPair
from pymusiclooper.console import rich_console, STYLE_SCORE_HIGH, STYLE_SCORE_MED, STYLE_SCORE_LOW
from pymusiclooper.core import MusicLooper
from pymusiclooper.exceptions import AudioLoadError, LoopNotFoundError
from pymusiclooper.utils import DEFAULT_OUTPUT_DIRECTORY_NAME


def _score_style(score: float) -> str:
    """Get Rich color style based on score value."""
    if score >= 0.75:
        return "bold green"
    elif score >= 0.5:
        return "yellow"
    else:
        return "red"


class LoopHandler:
    def __init__(
        self,
        *,
        path: str,
        min_duration_multiplier: float,
        min_loop_duration: Optional[float] = None,
        max_loop_duration: Optional[float] = None,
        approx_loop_position: Optional[tuple] = None,
        brute_force: bool = False,
        disable_pruning: bool = False,
        _progressbar: Progress = None,
        **kwargs,
    ):
        if approx_loop_position is not None:
            self.approx_loop_start = approx_loop_position[0]
            self.approx_loop_end = approx_loop_position[1]
        else:
            self.approx_loop_start = None
            self.approx_loop_end = None

        self.filepath = path
        self._musiclooper = MusicLooper(filepath=path)

        logging.info(f"Loaded \"{path}\". Analyzing...")

        self.loop_pair_list = self.musiclooper.find_loop_pairs(
            min_duration_multiplier=min_duration_multiplier,
            min_loop_duration=min_loop_duration,
            max_loop_duration=max_loop_duration,
            approx_loop_start=self.approx_loop_start,
            approx_loop_end=self.approx_loop_end,
            brute_force=brute_force,
            disable_pruning=disable_pruning,
        )
        self.interactive_mode = "PML_INTERACTIVE_MODE" in os.environ
        self.in_samples = "PML_DISPLAY_SAMPLES" in os.environ
        self._progressbar = _progressbar

    def get_all_loop_pairs(self) -> List[LoopPair]:
        """
        Returns the discovered loop points of an audio file as a list of LoopPair objects
        """
        return self.loop_pair_list

    @property
    def musiclooper(self) -> MusicLooper:
        """Returns the handler's `MusicLooper` instance."""
        return self._musiclooper
    
    def format_time(self, samples: int, in_samples: bool = False):
        return samples if in_samples else self.musiclooper.samples_to_ftime(samples)

    def play_looping(self, loop_start: int, loop_end: int):
        self.musiclooper.play_looping(loop_start, loop_end)

    def choose_loop_pair(self, interactive_mode=False):
        index = 0
        if self.loop_pair_list and interactive_mode:
            with _hideprogressbar(self._progressbar):
                index = self.interactive_handler()

        return self.loop_pair_list[index]

    def interactive_handler(self, show_top=25):
        preview_looper = self.musiclooper
        total_candidates = len(self.loop_pair_list)
        filename = os.path.basename(self.filepath)
        
        # Header panel
        rich_console.print()
        rich_console.print(Panel(
            f"[bold white]{filename}[/]\n"
            f"[dim]Found {total_candidates} loop candidate(s)[/]",
            title="[bold cyan]Loop Analysis[/]",
            border_style="cyan",
        ))
        
        # Results table
        table = Table(
            title=f"Showing {min(show_top, total_candidates)}/{total_candidates} loops",
            box=ROUNDED,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Start", style="green", width=10)
        table.add_column("End", style="green", width=10)
        table.add_column("Duration", style="dim", width=10)
        table.add_column("Note Δ", style="yellow", justify="right", width=8)
        table.add_column("Loud Δ", style="blue", justify="right", width=8)
        table.add_column("Score", justify="right", width=8)

        for idx, pair in enumerate(self.loop_pair_list[:show_top]):
            start_time = (
                str(pair.loop_start)
                if self.in_samples
                else preview_looper.samples_to_ftime(pair.loop_start)
            )
            end_time = (
                str(pair.loop_end)
                if self.in_samples
                else preview_looper.samples_to_ftime(pair.loop_end)
            )
            length = (
                str(pair.loop_end - pair.loop_start)
                if self.in_samples
                else preview_looper.samples_to_ftime(pair.loop_end - pair.loop_start)
            )
            score = pair.score
            score_style = _score_style(score)
            
            table.add_row(
                str(idx),
                str(start_time),
                str(end_time),
                str(length),
                f"{pair.note_distance:.3f}",
                f"{pair.loudness_difference:.3f}",
                f"[{score_style}]{score:.1%}[/]",
            )

        rich_console.print(table)
        
        # Command hints
        if show_top < total_candidates:
            rich_console.print()
            cmd_table = Table(show_header=False, box=None, padding=(0, 2))
            cmd_table.add_column("Key", style="cyan bold", width=10)
            cmd_table.add_column("Action", style="dim")
            cmd_table.add_row("NUMBER", "Select loop (e.g., 0)")
            cmd_table.add_row("NUMBERp", "Preview loop (e.g., 0p)")
            cmd_table.add_row("more", "Show more loops")
            cmd_table.add_row("all", "Show all loops")
            cmd_table.add_row("reset", "Reset to default view")
            rich_console.print(cmd_table)
        rich_console.print()

        def get_user_input():
            try:
                num_input = rich_console.input("[cyan bold]❯[/] ").strip()
                idx = 0
                preview = False

                if num_input == "more":
                    return self.interactive_handler(show_top=show_top * 2)
                if num_input == "all":
                    return self.interactive_handler(show_top=total_candidates)
                if num_input == "reset":
                    return self.interactive_handler()

                if num_input.endswith("p"):
                    idx = int(num_input[:-1])
                    preview = True
                else:
                    idx = int(num_input)

                if not 0 <= idx < len(self.loop_pair_list):
                    raise IndexError

                if preview:
                    pair = self.loop_pair_list[idx]
                    dur_s = preview_looper.samples_to_seconds(pair.loop_end - pair.loop_start)
                    rich_console.print(Panel(
                        f"[bold]Loop #{idx}[/]\n"
                        f"Start: [green]{preview_looper.samples_to_ftime(pair.loop_start)}[/] → "
                        f"End: [green]{preview_looper.samples_to_ftime(pair.loop_end)}[/]\n"
                        f"Duration: {dur_s:.1f}s │ Score: [{_score_style(pair.score)}]{pair.score:.1%}[/]",
                        title="[bold green]▶ Preview[/]",
                        subtitle="[dim]Ctrl+C to stop[/]",
                        border_style="green",
                    ))
                    loop_start = pair.loop_start
                    loop_end = pair.loop_end
                    # start preview 5 seconds before the looping point
                    offset = preview_looper.seconds_to_samples(5)
                    preview_offset = loop_end - offset if loop_end - offset > 0 else 0
                    preview_looper.play_looping(loop_start, loop_end, start_from=preview_offset)
                    return get_user_input()
                else:
                    return idx

            except (ValueError, IndexError):
                rich_console.print(f"[red]✗[/] Enter a number in range 0-{len(self.loop_pair_list)-1}")
                return get_user_input()
            except KeyboardInterrupt:
                rich_console.print("\n[dim]Stopped[/]")
                return get_user_input()

        try:
            selected_index = get_user_input()

            if selected_index is None:
                rich_console.print("[red]✗[/] Please select a valid number")
                return get_user_input()

            return selected_index
        except KeyboardInterrupt:
            rich_console.print("\n[yellow]⚠[/] Operation cancelled")
            sys.exit()


class LoopExportHandler(LoopHandler):
    def __init__(
        self,
        *,
        path: str,
        min_duration_multiplier: float,
        output_dir: str,
        min_loop_duration: Optional[float] = None,
        max_loop_duration: Optional[float] = None,
        approx_loop_position: Optional[tuple] = None,
        brute_force: bool = False,
        disable_pruning: bool = False,
        split_audio: bool = False,
        format: Literal["WAV", "FLAC", "OGG", "MP3"] = "WAV",
        to_txt: bool = False,
        to_stdout: bool = False,
        fmt: Literal["SAMPLES", "SECONDS", "TIME"] = "SAMPLES",
        alt_export_top: int = 0,
        tag_names: Optional[Tuple[str, str]] = None,
        tag_offset: Optional[bool] = None,
        batch_mode: bool = False,
        extended_length: float = 0,
        fade_length: float = 0,
        disable_fade_out: bool = False,
        **kwargs,
    ):
        super().__init__(
            path=path,
            min_duration_multiplier=min_duration_multiplier,
            min_loop_duration=min_loop_duration,
            max_loop_duration=max_loop_duration,
            approx_loop_position=approx_loop_position,
            brute_force=brute_force,
            disable_pruning=disable_pruning,
            **kwargs,
        )
        self.output_directory = output_dir
        self.split_audio = split_audio
        self.format = format
        self.to_txt = to_txt
        self.to_stdout = to_stdout
        self.fmt = fmt.lower()
        self.alt_export_top = alt_export_top
        self.tag_names = tag_names
        self.tag_offset = tag_offset
        self.batch_mode = batch_mode
        self.extended_length = extended_length
        self.disable_fade_out = disable_fade_out
        self.fade_length = fade_length
        self._is_autocreated_outdir = False

    def run(self):
        self.loop_pair_list = self.get_all_loop_pairs()
        chosen_loop_pair = self.choose_loop_pair(self.interactive_mode)
        loop_start = chosen_loop_pair.loop_start
        loop_end = chosen_loop_pair.loop_end

        # Runners that do not need an output directory
        if self.to_stdout:
            self.stdout_export_runner(loop_start, loop_end)

        # TODO: refactor into a context manager instead
        try:
            # Runners that need an output directory
            if (
                self.tag_names
                or self.to_txt
                or self.split_audio
                or self.extended_length
            ) and not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)
                self._is_autocreated_outdir = True

            if self.tag_names is not None:
                self.tag_runner(loop_start, loop_end)

            if self.to_txt:
                self.txt_export_runner(loop_start, loop_end)

            if self.split_audio:
                self.split_audio_runner(loop_start, loop_end)

            if self.extended_length:
                self.extend_track_runner(loop_start, loop_end)
        finally:
            if (
                self._is_autocreated_outdir
                and os.path.exists(self.output_directory)
                and len(os.listdir(self.output_directory)) == 0
            ):
                os.rmdir(self.output_directory)

    def split_audio_runner(self, loop_start: int, loop_end: int):
        try:
            self.musiclooper.export(
                loop_start,
                loop_end,
                format=self.format,
                output_dir=self.output_directory
            )
            if self.batch_mode:
                logging.info(f"Exported \"{self.musiclooper.filename}\" to \"{self.output_directory}\"")
            else:
                rich_console.print(Panel(
                    f"[bold green]✓ Export complete[/]\n\n"
                    f"[dim]File:[/] {self.musiclooper.filename}\n"
                    f"[dim]Output:[/] {self.output_directory}\n"
                    f"[dim]Format:[/] {self.format}\n"
                    f"[dim]Sections:[/] intro, loop, outro",
                    title="[bold cyan]Split Audio[/]",
                    border_style="green",
                ))
        # Usually: unknown file format specified; raised by soundfile
        except ValueError as e:
            logging.error(e)

    def extend_track_runner(self, loop_start: int, loop_end: int):
        # Add a progress bar since it could take some time to export
        # Do not enable if batch mode is active, since it already has a progress bar
        progress = None
        if not self.batch_mode:
            progress = Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=rich_console,
                transient=True,
            )
            progress.add_task(f"[cyan]Extending {self.musiclooper.filename}...", total=None)
            progress.start()
        try:
            output_path = self.musiclooper.extend(
                loop_start,
                loop_end,
                format=self.format,
                output_dir=self.output_directory,
                extended_length=self.extended_length,
                disable_fade_out=self.disable_fade_out,
                fade_length=self.fade_length,
            )
            if self.batch_mode:
                logging.info(f"Extended \"{self.musiclooper.filename}\" → \"{output_path}\"")
            else:
                progress.stop()
                fade_info = "disabled" if self.disable_fade_out else f"{self.fade_length}s"
                rich_console.print(Panel(
                    f"[bold green]✓ Extended track created[/]\n\n"
                    f"[dim]Source:[/] {self.musiclooper.filename}\n"
                    f"[dim]Output:[/] {output_path}\n"
                    f"[dim]Length:[/] {self.extended_length}s\n"
                    f"[dim]Fade out:[/] {fade_info}",
                    title="[bold cyan]Extend Track[/]",
                    border_style="green",
                ))
        # Usually: unknown file format specified; raised by soundfile
        except ValueError as e:
            if progress:
                progress.stop()
            logging.error(e)

    def txt_export_runner(self, loop_start: int, loop_end: int):
        if self.alt_export_top != 0:
            self.alt_export_runner(mode="TXT")
        else:
            self.musiclooper.export_txt(
                self._fmt(loop_start),
                self._fmt(loop_end),
                output_dir=self.output_directory,
            )
            out_path = os.path.join(self.output_directory, "loop.txt")
            if self.batch_mode:
                logging.info(f"Exported loop points for \"{self.musiclooper.filename}\"")
            else:
                rich_console.print(f"[green]✓[/] Exported loop points → [cyan]{out_path}[/]")

    def stdout_export_runner(self, loop_start: int, loop_end: int):
        if self.alt_export_top != 0:
            self.alt_export_runner(mode="STDOUT")
        else:
            rich_console.print(Panel(
                f"[bold]LOOP_START[/] = [green]{self._fmt(loop_start)}[/]\n"
                f"[bold]LOOP_END[/]   = [green]{self._fmt(loop_end)}[/]",
                title=f"[bold cyan]Loop Points: {self.musiclooper.filename}[/]",
                border_style="cyan",
            ))

    def alt_export_runner(self, mode: Literal["STDOUT", "TXT"]):
        pair_list_slice = (
            self.loop_pair_list
            if self.alt_export_top < 0
            or self.alt_export_top >= len(self.loop_pair_list)
            else self.loop_pair_list[: self.alt_export_top]
        )

        def fmt_line(pair: LoopPair):
            return f"{self._fmt(pair.loop_start)} {self._fmt(pair.loop_end)} {pair.note_distance} {pair.loudness_difference} {pair.score}\n"

        formatted_lines = [fmt_line(pair) for pair in pair_list_slice]
        if mode == "STDOUT":
            rich_console.out(*formatted_lines, sep="", end="")
        elif mode == "TXT":
            out_path = os.path.join(
                self.output_directory, f"{self.musiclooper.filename}.alt_export.txt"
            )
            with open(out_path, mode="w") as f:
                f.writelines(formatted_lines)

    def tag_runner(self, loop_start: int, loop_end: int):        
        loop_start_tag, loop_end_tag = self.tag_names
        loop_start, loop_end = self.musiclooper.export_tags(
            loop_start,
            loop_end,
            loop_start_tag,
            loop_end_tag,
            is_offset=self.tag_offset,
            output_dir=self.output_directory,
        )
        if self.batch_mode:
            logging.info(f"Tagged \"{self.musiclooper.filename}\" → {self.output_directory}")
        else:
            rich_console.print(Panel(
                f"[bold green]✓ Tags written[/]\n\n"
                f"[cyan]{loop_start_tag}[/] = [green]{loop_start}[/]\n"
                f"[cyan]{loop_end_tag}[/] = [green]{loop_end}[/]\n\n"
                f"[dim]Output:[/] {self.output_directory}",
                title=f"[bold cyan]Tag: {self.musiclooper.filename}[/]",
                border_style="green",
            ))

    def _fmt(self, samples: int):
        if self.fmt == "seconds":
            return str(self.musiclooper.samples_to_seconds(samples))
        elif self.fmt == "time":
            return str(self.musiclooper.samples_to_ftime(samples))
        else:
            return str(samples)


class BatchHandler:
    def __init__(
        self,
        *,
        path: str,
        output_dir: str,
        recursive: bool = False,
        flatten: bool = False,
        **kwargs,
    ):
        """Processes all audio files in a directory with `LoopExportHandler`.

        Args:
            path (str): Path to directory.
            output_dir (str): Output directory to use for exports. 
            recursive (bool, optional): Process directories recursively. Defaults to False.
            flatten (bool, optional): Flatten the output directory structure instead of preserving it when processing it recursively. Defaults to False.
            kwargs: Additional `kwargs` are passed onto `LoopExportHandler`.
        """
        self.directory_path = os.path.abspath(path)
        self.output_directory = output_dir
        self.recursive = recursive
        self.flatten = flatten
        self.kwargs = kwargs
        self._created_dirs = []

    def run(self):
        files = self.get_files_in_directory(
            self.directory_path, recursive=self.recursive
        )

        if len(files) == 0:
            raise FileNotFoundError(f"No files found in \"{self.directory_path}\"")

        output_dirs = (
            None
            if self.flatten
            else self.clone_file_tree_structure(files, self.output_directory)
        )

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            console=rich_console,
        ) as progress:
            pbar = progress.add_task("Processing...", total=len(files))
            for file_idx, file_path in enumerate(files):
                progress.update(
                    pbar,
                    advance=1,
                    description=(
                        f"Processing \"{os.path.relpath(file_path, self.directory_path)}\""
                    ),
                )
                task_kwargs = {
                    **self.kwargs,
                    "_progressbar": progress,
                    "path": file_path,
                    "output_dir": self.output_directory if self.flatten else output_dirs[file_idx]
                }
                try:
                    self._batch_export_helper(**task_kwargs)
                finally:
                    self._cleanup_empty_created_dirs()

    def clone_file_tree_structure(self, in_files: List[str], output_directory: str) -> List[str]:
        common_path = os.path.commonpath(in_files)
        output_dirs = [
            os.path.join(
                os.path.abspath(output_directory),
                os.path.dirname(os.path.relpath(file, start=common_path)),
            )
            for file in in_files
        ]
        for out_dir in output_dirs:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                self._created_dirs.append(out_dir)
        return output_dirs

    @staticmethod
    def get_files_in_directory(dir_path: str, recursive: bool = False) -> List[str]:
        return (
            [
                os.path.join(directory, filename)
                for directory, sub_dir_list, file_list in os.walk(dir_path)
                for filename in file_list
            ]
            if recursive
            else [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
            ]
        )

    @staticmethod
    def _batch_export_helper(**kwargs):
        try:
            export_handler = LoopExportHandler(**kwargs, batch_mode=True)
            export_handler.run()
        except (AudioLoadError, LoopNotFoundError) as e:
            logging.error(e)
        except Exception as e:
            logging.error(e)

    def _cleanup_empty_created_dirs(self):
        dirs_to_check = self._created_dirs

        if os.path.basename(self.output_directory) == DEFAULT_OUTPUT_DIRECTORY_NAME:
            dirs_to_check += [self.output_directory]

        for directory in dirs_to_check:
            if (
                os.path.exists(directory)
                and len(os.listdir(directory)) == 0
            ):
                os.removedirs(directory)


@contextmanager
def _hideprogressbar(progress: Progress):
    """
    Intended to pause and hide the progress bar while a prompt is active.

    Based on @abrahammurciano's answer in rich issue #1535
    https://github.com/Textualize/rich/issues/1535#issuecomment-1745297594
    """
    # Handle edge case where a progressbar might not exist
    if progress is None:
        try:
            yield
        finally:
            pass
        return
    transient = progress.live.transient # save the old value
    progress.live.transient = True
    progress.stop()
    try:
        yield
    finally:
        # make space for the progress to use so it doesn't overwrite any previous lines
        print("\n" * (len(progress.tasks) - 2))
        progress.live.transient = transient # restore the old value
        progress.start()
