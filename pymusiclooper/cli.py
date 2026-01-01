import functools
import logging
import os
import tempfile
import warnings

import rich_click as click
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from rich.traceback import install as rich_traceback_handler
from rich_click.patch import patch as rich_click_patch
from yt_dlp.utils import YoutubeDLError

rich_click_patch()
from click_option_group import RequiredMutuallyExclusiveOptionGroup, optgroup
from click_params import URL as UrlParamType

from pymusiclooper import __version__
from pymusiclooper.analysis import extract_features
from pymusiclooper.multihop import MultiHopLoop, find_multi_hop_loops
from pymusiclooper.console import _COMMAND_GROUPS, _OPTION_GROUPS, rich_console
from pymusiclooper.core import MusicLooper
from pymusiclooper.exceptions import AudioLoadError, LoopNotFoundError
from pymusiclooper.handler import BatchHandler, LoopExportHandler, LoopHandler
from pymusiclooper.utils import download_audio, get_outputdir, mk_outputdir

# CLI --help styling
click.rich_click.OPTION_GROUPS = _OPTION_GROUPS
click.rich_click.COMMAND_GROUPS = _COMMAND_GROUPS
click.rich_click.USE_RICH_MARKUP = True
# End CLI styling


@click.group("pymusiclooper", epilog="Full documentation and examples can be found at https://github.com/arkrow/PyMusicLooper")
@click.option("--debug", "-d", is_flag=True, default=False, help="Enables debugging mode.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enables verbose logging output.")
@click.option("--interactive", "-i", is_flag=True, default=False, help="Enables interactive mode to manually preview/choose the desired loop point.")
@click.option("--samples", "-s", is_flag=True, default=False, help="Display all the loop points shown in interactive mode in sample points instead of the default mm:ss.sss format.")
@click.version_option(__version__, prog_name="pymusiclooper", message="%(prog)s %(version)s")
def cli_main(debug, verbose, interactive, samples):
    """A program for repeating music seamlessly and endlessly, by automatically finding the best loop points."""
    # Store flags in environ instead of passing them as parameters
    if debug:
        os.environ["PML_DEBUG"] = "1"
        warnings.simplefilter("default")
        rich_traceback_handler(console=rich_console, suppress=[click])
    else:
        warnings.filterwarnings("ignore")

    if verbose:
        os.environ["PML_VERBOSE"] = "1"
    if interactive:
        os.environ["PML_INTERACTIVE_MODE"] = "1"
    if samples:
        os.environ["PML_DISPLAY_SAMPLES"] = "1"

    if verbose:
        logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=[RichHandler(level=logging.INFO, console=rich_console, rich_tracebacks=True, show_path=debug, show_time=False, tracebacks_suppress=[click])])
    else:
        logging.basicConfig(format="%(message)s", level=logging.ERROR, handlers=[RichHandler(level=logging.ERROR, console=rich_console, show_time=False, show_path=False)])


def common_path_options(f):
    @optgroup.group("audio path", cls=RequiredMutuallyExclusiveOptionGroup, help="the path to the audio track(s) to load")
    @optgroup.option("--path", type=click.Path(exists=True), default=None, help=r"Path to the audio file(s). [dim cyan]\[mutually exclusive with --url][/] [dim red]\[at least one required][/]")
    @optgroup.option("--url",type=UrlParamType, default=None, help=r"Link to the youtube video (or any stream supported by yt-dlp) to extract audio from and use. [dim cyan]\[mutually exclusive with --path][/] [dim red]\[at least one required][/]")

    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options


def common_loop_options(f):
    @click.option('--min-duration-multiplier', type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True), default=0.35, show_default=True, help="The minimum loop duration as a multiplier of the audio track's total duration.")
    @click.option('--min-loop-duration', type=click.FloatRange(min=0, min_open=True), default=None, help='The minimum loop duration in seconds. [dim](overrides --min-duration-multiplier if set)[/]')
    @click.option('--max-loop-duration', type=click.FloatRange(min=0, min_open=True), default=None, help='The maximum loop duration in seconds.')
    @click.option('--approx-loop-position', type=click.FloatRange(min=0), nargs=2, default=None, help='The approximate desired loop start and loop end in seconds. [dim]([cyan]+/-2[/] second search window for each point)[/]')
    @click.option("--brute-force", is_flag=True, default=False, help=r"Check the entire audio track instead of just the detected beats. [dim yellow](Warning: may take several minutes to complete.)[/]")
    @click.option("--disable-pruning", is_flag=True, default=False, help="Disables filtering of the detected loop points from the initial pass.")

    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options


def common_export_options(f):
    @click.option('--output-dir', '-o', type=click.Path(exists=False, writable=True, file_okay=False), help="The output directory to use for the exported files.")
    @click.option("--recursive", "-r", is_flag=True, default=False, help="Process directories recursively.")
    @click.option("--flatten", "-f", is_flag=True, default=False, help="Flatten the output directory structure instead of preserving it when using the --recursive flag. [dim yellow](Note: files with identical filenames are silently overwritten.)[/]")
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options


@cli_main.command()
@common_path_options
@common_loop_options
def play(**kwargs):
    """Play an audio file on repeat from the terminal with the best discovered loop points, or a chosen point if interactive mode is active."""
    try:
        if kwargs.get("url", None) is not None:
            kwargs["path"] = download_audio(kwargs["url"], tempfile.gettempdir())

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=rich_console,
            transient=True
        ) as progress:
            progress.add_task("Processing", total=None)
            handler = LoopHandler(**kwargs)

        in_samples = "PML_DISPLAY_SAMPLES" in os.environ
        interactive_mode = "PML_INTERACTIVE_MODE" in os.environ

        try:
            chosen_loop_pair = handler.choose_loop_pair(interactive_mode=interactive_mode)
        except KeyboardInterrupt:
            # User exited interactive mode
            return

        start_time = handler.format_time(chosen_loop_pair.loop_start, in_samples=in_samples)
        end_time = handler.format_time(chosen_loop_pair.loop_end, in_samples=in_samples)

        rich_console.print(
            "\nPlaying with looping active from [green]{}[/] back to [green]{}[/]; similarity: {:.2%}".format(
                end_time,
                start_time,
                chosen_loop_pair.score,
            )
        )
        rich_console.print("(Press [red]Ctrl+C[/] to stop looping.)")
        
        # Start 5 seconds before loop end to hear the transition
        preview_offset = handler.musiclooper.seconds_to_samples(5)
        start_from = max(0, chosen_loop_pair.loop_end - preview_offset)
        handler.play_looping(chosen_loop_pair.loop_start, chosen_loop_pair.loop_end, start_from=start_from)

    except YoutubeDLError:
        # Already logged from youtube.py
        pass
    except (AudioLoadError, LoopNotFoundError, Exception) as e:
        print_exception(e)


@cli_main.command()
@click.option('--path', type=click.Path(exists=True), required=True, help='Path to the audio file.')
@click.option("--tag-names", type=str, required=False, nargs=2, help="Name of the loop metadata tags to read from, e.g. --tag-names LOOP_START LOOP_END  (note: values must be integers and in sample units). Default: auto-detected.")
@click.option("--tag-offset/--no-tag-offset", is_flag=True, default=None, help="Always parse second loop metadata tag as a relative length / or as an absolute length. Default: auto-detected based on tag name.")
def play_tagged(path, tag_names, tag_offset):
    """Skips loop analysis and reads the loop points directly from the tags present in the file."""
    try:
        if tag_names is None:
            tag_names = [None, None]

        looper = MusicLooper(path)
        loop_start, loop_end = looper.read_tags(tag_names[0], tag_names[1], tag_offset)

        in_samples = "PML_DISPLAY_SAMPLES" in os.environ

        start_time = (
            loop_start
            if in_samples
            else looper.samples_to_ftime(loop_start)
        )
        end_time = (
            loop_end
            if in_samples
            else looper.samples_to_ftime(loop_end)
        )

        rich_console.print(f"\nPlaying with looping active from [green]{end_time}[/] back to [green]{start_time}[/]")
        rich_console.print("(Press [red]Ctrl+C[/] to stop looping.)")
        
        # Start 5 seconds before loop end to hear the transition
        preview_offset = looper.seconds_to_samples(5)
        start_from = max(0, loop_end - preview_offset)
        looper.play_looping(loop_start, loop_end, start_from=start_from)

    except Exception as e:
        print_exception(e)


@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option('--format', type=click.Choice(("WAV", "FLAC", "OGG", "MP3"), case_sensitive=False), default="WAV", show_default=True, help="Audio format to use for the exported split audio files.")
def split_audio(**kwargs):
    """Split the input audio into intro, loop and outro sections."""
    kwargs["split_audio"] = True
    run_handler(**kwargs)

@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option('--format', type=click.Choice(("WAV", "FLAC", "OGG", "MP3"), case_sensitive=False), default="MP3", show_default=True, help="Audio format to use for the output audio file.")
@click.option('--extended-length', type=float, required=True, help="Desired length of the extended looped track in seconds. [Must be longer than the audio's original length.]")
@click.option('--fade-length', type=float, default=5, show_default=True, help="Desired length of the loop fade out in seconds.")
@click.option('--disable-fade-out', is_flag=True, default=False, help="Extend the track with all its sections (intro/loop/outro) without fading out. --extended-length will be treated as an 'at least' constraint.")
def extend(**kwargs):
    """Create an extended version of the input audio by looping it to a specific length."""
    run_handler(**kwargs)


@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option("--export-to", type=click.Choice(("STDOUT", "TXT"), case_sensitive=False), default="STDOUT", show_default=True, help="STDOUT: print the loop points of a track in samples to the terminal; TXT: export the loop points of a track in samples and append to a loop.txt file.")
@click.option("--fmt", type=click.Choice(("SAMPLES", "SECONDS", "TIME"), case_sensitive=False), default="SAMPLES", show_default=True, help="Export loop points formatted as samples (default), seconds, or time (mm:ss.sss).")
@click.option("--alt-export-top", type=int, default=0, help="Alternative export format of the top N loop points instead of the best detected/chosen point. --alt-export-top -1 to export all points.")
def export_points(**kwargs):
    """Export the best discovered or chosen loop points to a text file or to the terminal."""
    kwargs["to_stdout"] = kwargs["export_to"].upper() == "STDOUT"
    kwargs["to_txt"] = kwargs["export_to"].upper() == "TXT"
    kwargs.pop("export_to", "")

    run_handler(**kwargs)


@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option('--tag-names', type=str, required=True, nargs=2, help='Name of the loop metadata tags to use, e.g. --tag-names LOOP_START LOOP_END')
@click.option("--tag-offset/--no-tag-offset", is_flag=True, default=None, help="Always export second loop metadata tag as a relative length / or as an absolute length. Default: auto-detected based on tag name.")
def tag(**kwargs):
    """Adds metadata tags of loop points to a copy of the input audio file(s)."""
    run_handler(**kwargs)


@cli_main.command()
@click.option('--path', type=click.Path(exists=True), required=True, help='Path to the audio file.')
@click.option('--n-hops', type=click.IntRange(min=2, max=10), default=2, show_default=True, help='Number of transition points in the loop chain. 2=A->B->A, 3=A->B->C->A, etc.')
@click.option('--min-segment-duration', type=click.FloatRange(min=1.0), default=4.0, show_default=True, help='Minimum segment duration in seconds.')
@click.option('--max-segment-duration', type=click.FloatRange(min=1.0), default=None, help='Maximum segment duration in seconds.')
def multi_hop(**kwargs):
    """Find multi-hop loop chains with multiple seamless transitions.
    
    Instead of one loop point (start -> end -> start), finds a chain of
    segments that transition seamlessly into each other.
    
    Example with n_hops=2: Play intro, then loop A->B->A repeatedly.
    
    Example with n_hops=3: Play intro, then loop A->B->C->A repeatedly.
    """
    try:
        path = kwargs["path"]
        n_hops = kwargs["n_hops"]
        min_seg = kwargs["min_segment_duration"]
        max_seg = kwargs["max_segment_duration"]
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=rich_console,
            transient=True
        ) as progress:
            task = progress.add_task("Extracting features...", total=None)
            looper = MusicLooper(path)
            from pymusiclooper.analysis import extract_features
            features = extract_features(looper.mlaudio, None, None, False)
            
            progress.update(task, description="Finding similar sections...")
            chains = find_multi_hop_loops(
                looper.mlaudio,
                features,
                n_hops=n_hops,
                min_segment_duration=min_seg,
                max_segment_duration=max_seg,
            )
        
        if not chains:
            rich_console.print("[yellow]No multi-hop chains found. Try adjusting parameters or use single-loop mode.[/]")
            return
        
        # Display results
        in_samples = "PML_DISPLAY_SAMPLES" in os.environ
        
        rich_console.print(f"\n[bold]Multi-hop chains for \"{os.path.basename(path)}\":[/]\n")
        
        table = Table(title=f"Top {min(10, len(chains))} chains (n_hops={n_hops})")
        table.add_column("Index", justify="right", style="cyan")
        table.add_column("Segments", style="magenta")
        table.add_column("Transitions", style="green")
        table.add_column("Score", justify="right", style="red")
        
        for idx, chain in enumerate(chains[:10]):
            seg_strs = []
            for start_s, end_s in chain.segment_samples:
                if in_samples:
                    seg_strs.append(f"{start_s}->{end_s}")
                else:
                    seg_strs.append(f"{looper.samples_to_ftime(start_s)}->{looper.samples_to_ftime(end_s)}")
            
            trans_strs = [f"{t:.2f}" for t in chain.transitions]
            
            table.add_row(
                str(idx),
                " | ".join(seg_strs),
                ", ".join(trans_strs),
                f"{chain.total_score:.2%}",
            )
        
        rich_console.print(table)
        
        # Show best chain details
        best = chains[0]
        rich_console.print(f"\n[bold green]Best chain (score: {best.total_score:.2%}):[/]")
        rich_console.print("Playback order:")
        for i, (start_s, end_s) in enumerate(best.segment_samples):
            if in_samples:
                rich_console.print(f"  Segment {i+1}: {start_s} -> {end_s}")
            else:
                rich_console.print(f"  Segment {i+1}: {looper.samples_to_ftime(start_s)} -> {looper.samples_to_ftime(end_s)}")
        rich_console.print(f"  [dim](Then loops back to Segment 1)[/]")
        
        # Interactive preview
        if "PML_INTERACTIVE_MODE" in os.environ:
            rich_console.print("\n[bold cyan]Interactive Mode[/]")
            rich_console.print("[dim]Commands:[/]")
            rich_console.print("  [cyan]<number>[/] - Preview chain (e.g. 0)")
            rich_console.print("  [cyan]p<number>[/] - Preview with info (e.g. p0)")
            rich_console.print("  [cyan]more[/] - Show more chains")
            rich_console.print("  [cyan]q[/] or [cyan]exit[/] - Quit\n")
            
            show_top = 10
            while True:
                try:
                    user_input = rich_console.input("[bold]> [/]").strip()
                    
                    if user_input.lower() in ('q', 'quit', 'exit'):
                        break
                    elif user_input.lower() == 'more':
                        show_top = min(len(chains), show_top + 10)
                        # Re-display table
                        table = Table(title=f"Top {show_top} chains (n_hops={n_hops})")
                        table.add_column("Index", justify="right", style="cyan")
                        table.add_column("Segments", style="magenta")
                        table.add_column("Transitions", style="green")
                        table.add_column("Score", justify="right", style="red")
                        
                        for idx, chain in enumerate(chains[:show_top]):
                            seg_strs = []
                            for start_s, end_s in chain.segment_samples:
                                if in_samples:
                                    seg_strs.append(f"{start_s}->{end_s}")
                                else:
                                    seg_strs.append(f"{looper.samples_to_ftime(start_s)}->{looper.samples_to_ftime(end_s)}")
                            
                            trans_strs = [f"{t:.2f}" for t in chain.transitions]
                            table.add_row(str(idx), " | ".join(seg_strs), ", ".join(trans_strs), f"{chain.total_score:.2%}")
                        
                        rich_console.print(table)
                        continue
                    
                    preview = user_input.lower().startswith('p')
                    if preview:
                        user_input = user_input[1:]
                    
                    idx = int(user_input)
                    if 0 <= idx < len(chains):
                        chain = chains[idx]
                        
                        if preview:
                            rich_console.print(f"\n[bold]Chain #{idx} Details:[/]")
                            rich_console.print(f"  Score: [green]{chain.total_score:.2%}[/]")
                            rich_console.print(f"  Segments: [cyan]{len(chain.segment_samples)}[/]")
                            rich_console.print(f"  Transitions: {', '.join([f'{t:.2f}' for t in chain.transitions])}")
                            rich_console.print(f"\n  [bold]Playback order:[/]")
                            for i, (start_s, end_s) in enumerate(chain.segment_samples):
                                dur = looper.samples_to_seconds(end_s - start_s)
                                if in_samples:
                                    rich_console.print(f"    Seg{i+1}: {start_s} -> {end_s} ({dur:.2f}s)")
                                else:
                                    rich_console.print(f"    Seg{i+1}: {looper.samples_to_ftime(start_s)} -> {looper.samples_to_ftime(end_s)} ({dur:.2f}s)")
                            rich_console.print(f"    [dim]→ Loop back to Seg1[/]\n")
                        
                        rich_console.print(f"[bold green]Playing chain #{idx}...[/] [dim](Press Ctrl+C to stop)[/]")
                        try:
                            # Play all segments in sequence, looping
                            looper.play_multi_hop(chain.segment_samples)
                        except KeyboardInterrupt:
                            rich_console.print("\n[dim]Stopped playback[/]")
                            # Continue loop to ask for next input
                            continue
                    else:
                        rich_console.print(f"[red]Index must be 0-{len(chains)-1}[/]")
                except ValueError:
                    rich_console.print("[red]Invalid input. Enter a number, 'p<number>', 'more', 'q', or 'exit'[/]")
                except KeyboardInterrupt:
                    rich_console.print("\n[dim]Exiting...[/]")
                    break
                    
    except (AudioLoadError, LoopNotFoundError, Exception) as e:
        print_exception(e)


@cli_main.command()
@click.option('--path', type=click.Path(exists=True), required=True, help='Path to the audio file.')
@click.option('--n-hops', type=click.IntRange(min=2, max=10), default=2, show_default=True, help='Number of transition points.')
@click.option('--min-segment-duration', type=click.FloatRange(min=1.0), default=4.0, show_default=True, help='Minimum segment duration in seconds.')
@click.option('--max-segment-duration', type=click.FloatRange(min=1.0), default=None, help='Maximum segment duration in seconds.')
@click.option('--output-dir', '-o', type=click.Path(exists=False, writable=True, file_okay=False), help="Output directory for the export.")
@click.option('--top', type=int, default=1, show_default=True, help='Export top N chains.')
def export_multi_hop(**kwargs):
    """Export multi-hop loop chain points to a text file."""
    try:
        path = kwargs["path"]
        n_hops = kwargs["n_hops"]
        min_seg = kwargs["min_segment_duration"]
        max_seg = kwargs["max_segment_duration"]
        output_dir = kwargs.get("output_dir") or os.path.dirname(path) or "."
        top_n = kwargs["top"]
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=rich_console,
            transient=True
        ) as progress:
            task = progress.add_task("Extracting features...", total=None)
            looper = MusicLooper(path)
            from pymusiclooper.analysis import extract_features
            features = extract_features(looper.mlaudio, None, None, False)
            
            progress.update(task, description="Finding similar sections...")
            chains = find_multi_hop_loops(
                looper.mlaudio,
                features,
                n_hops=n_hops,
                min_segment_duration=min_seg,
                max_segment_duration=max_seg,
            )
        
        if not chains:
            rich_console.print("[yellow]No multi-hop chains found.[/]")
            return
        
        # Export to file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{filename}_multihop.txt")
        
        with open(out_path, "w") as f:
            f.write(f"# Multi-hop loop chains for: {os.path.basename(path)}\n")
            f.write(f"# n_hops: {n_hops}\n")
            f.write(f"# Format: chain_index score segment1_start segment1_end [segment2_start segment2_end ...]\n\n")
            
            for idx, chain in enumerate(chains[:top_n]):
                parts = [str(idx), f"{chain.total_score:.4f}"]
                for start_s, end_s in chain.segment_samples:
                    parts.extend([str(start_s), str(end_s)])
                f.write(" ".join(parts) + "\n")
        
        rich_console.print(f"[green]Exported {min(top_n, len(chains))} multi-hop chains to:[/] {out_path}")
        
        # Also print best chain to stdout
        best = chains[0]
        rich_console.print(f"\n[bold]Best chain (score: {best.total_score:.2%}):[/]")
        for i, (start_s, end_s) in enumerate(best.segment_samples):
            rich_console.print(f"  Segment {i+1}: {start_s} -> {end_s} samples")
            rich_console.print(f"            ({looper.samples_to_ftime(start_s)} -> {looper.samples_to_ftime(end_s)})")
            
    except (AudioLoadError, LoopNotFoundError, Exception) as e:
        print_exception(e)


@cli_main.command()
@click.option('--path', type=click.Path(exists=True), required=True, help='Path to the audio file.')
@click.option('--start', type=float, required=True, help='Start of the segment in seconds.')
@click.option('--end', type=float, required=True, help='End of the segment in seconds.')
@click.option('--mode', type=click.Choice(['internal', 'boundary', 'extended']), default='boundary', show_default=True, help='Search mode: internal (within segment), boundary (optimize edges), extended (allow extension).')
@click.option('--top', type=int, default=5, show_default=True, help='Show top N results.')
def segment_loop(**kwargs):
    """Find optimal loop points for a user-selected audio segment.
    
    When you have a favorite moment in a track and want to loop it perfectly,
    this command analyzes the segment and finds the best loop points.
    
    Search modes:
    
    \b
    - internal: Find repeating patterns WITHIN the segment
    - boundary: Optimize the start/end for seamless looping (default)
    - extended: Allow slight extension beyond the segment for better loops
    
    Example:
    
    \b
        pymusiclooper segment-loop --path song.mp3 --start 45.5 --end 78.2
    """
    try:
        path = kwargs["path"]
        start_sec = kwargs["start"]
        end_sec = kwargs["end"]
        mode = kwargs["mode"]
        top_n = kwargs["top"]
        
        if start_sec >= end_sec:
            rich_console.print("[red]Error: start must be less than end[/]")
            return
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=rich_console,
            transient=True
        ) as progress:
            task = progress.add_task("Analyzing segment...", total=None)
            
            looper = MusicLooper(path)
            
            progress.update(task, description="Extracting features...")
            from pymusiclooper.analysis import find_segment_loop_points, extract_features
            features = extract_features(looper.mlaudio, None, None, False)
            
            progress.update(task, description=f"Finding loops ({mode} mode)...")
            results = find_segment_loop_points(
                looper.mlaudio,
                features,
                start_sec,
                end_sec,
                search_mode=mode,
                n_results=top_n,
            )
        
        if not results:
            rich_console.print("[yellow]No suitable loops found for this segment.[/]")
            rich_console.print("[dim]Try different --mode or adjust segment boundaries.[/]")
            return
        
        in_samples = "PML_DISPLAY_SAMPLES" in os.environ
        
        # Display results
        rich_console.print(f"\n[bold]Segment Loop Analysis for \"{os.path.basename(path)}\"[/]")
        rich_console.print(f"Segment: [cyan]{start_sec:.2f}s[/] - [cyan]{end_sec:.2f}s[/] ({end_sec - start_sec:.2f}s)")
        rich_console.print(f"Mode: [green]{mode}[/]\n")
        
        table = Table(title=f"Top {len(results)} Loop Points")
        table.add_column("#", justify="right", style="cyan", width=3)
        table.add_column("Start", style="green")
        table.add_column("End", style="green")
        table.add_column("Duration", style="magenta")
        table.add_column("Score", justify="right", style="red")
        table.add_column("Type", style="yellow")
        table.add_column("Quality", style="dim")
        
        for idx, result in enumerate(results):
            if in_samples:
                start_str = str(result.loop_start)
                end_str = str(result.loop_end)
            else:
                start_str = looper.samples_to_ftime(result.loop_start)
                end_str = looper.samples_to_ftime(result.loop_end)
            
            duration = looper.samples_to_seconds(result.loop_end - result.loop_start)
            
            # Quality indicator
            if result.score > 0.8:
                quality = "★★★"
            elif result.score > 0.6:
                quality = "★★"
            else:
                quality = "★"
            
            table.add_row(
                str(idx),
                start_str,
                end_str,
                f"{duration:.2f}s",
                f"{result.score:.2%}",
                result.boundary_type,
                quality,
            )
        
        rich_console.print(table)
        
        # Best result details
        best = results[0]
        rich_console.print(f"\n[bold green]Best Loop (#{0}):[/]")
        rich_console.print(f"  {best.explanation}")
        rich_console.print(f"  Rhythm: {best.rhythm_alignment:.2%} | Harmony: {best.harmonic_match:.2%} | Transition: {best.transition_quality:.2%}")
        
        # Interactive mode
        if "PML_INTERACTIVE_MODE" in os.environ:
            rich_console.print("\n[bold cyan]Interactive Mode[/]")
            rich_console.print("[dim]Commands:[/]")
            rich_console.print("  [cyan]<number>[/] - Preview loop (e.g. 0)")
            rich_console.print("  [cyan]q[/] or [cyan]exit[/] - Quit\n")
            
            while True:
                try:
                    user_input = rich_console.input("[bold]> [/]").strip()
                    
                    if user_input.lower() in ('q', 'quit', 'exit'):
                        break
                    
                    idx = int(user_input)
                    if 0 <= idx < len(results):
                        result = results[idx]
                        rich_console.print(f"\n[green]Playing loop #{idx}...[/] [dim](Press Ctrl+C to stop)[/]")
                        rich_console.print(f"  {result.explanation}\n")
                        try:
                            # Start 5 seconds before loop end to hear the transition
                            preview_offset = looper.seconds_to_samples(5)
                            start_from = max(0, result.loop_end - preview_offset)
                            looper.play_looping(result.loop_start, result.loop_end, start_from=start_from)
                        except KeyboardInterrupt:
                            rich_console.print("\n[dim]Stopped playback[/]")
                            # Continue loop to ask for next input
                            continue
                    else:
                        rich_console.print(f"[red]Index must be 0-{len(results)-1}[/]")
                        
                except ValueError:
                    rich_console.print("[red]Invalid input. Enter a number, 'q', or 'exit'[/]")
                except KeyboardInterrupt:
                    rich_console.print("\n[dim]Exiting...[/]")
                    break
        
    except (AudioLoadError, LoopNotFoundError, Exception) as e:
        print_exception(e)


def run_handler(**kwargs):
    try:
        if kwargs.get("url", None) is not None:
            kwargs["output_dir"] = mk_outputdir(os.getcwd(), kwargs["output_dir"])
            kwargs["path"] = download_audio(kwargs["url"], kwargs["output_dir"])
        else:  
            kwargs["output_dir"] = get_outputdir(kwargs["path"], kwargs["output_dir"])

        if os.path.isfile(kwargs["path"]):
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=rich_console,
                transient=True
            ) as progress:
                progress.add_task("Processing", total=None)
                export_handler = LoopExportHandler(**kwargs)
            export_handler.run()
        else:
            batch_handler = BatchHandler(**kwargs)
            batch_handler.run()
    except YoutubeDLError:
        # Already logged from youtube.py
        pass
    except (AudioLoadError, LoopNotFoundError, Exception) as e:
        print_exception(e)

def print_exception(e: Exception):
    if "PML_DEBUG" in os.environ:
        rich_console.print_exception(suppress=[click])
    else:
        logging.error(e)

if __name__ == "__main__":
    cli_main()
