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

        chosen_loop_pair = handler.choose_loop_pair(interactive_mode=interactive_mode)

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

        handler.play_looping(chosen_loop_pair.loop_start, chosen_loop_pair.loop_end)

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

        looper.play_looping(loop_start, loop_end)

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
@click.option('--n-hops', type=click.IntRange(min=2, max=10), default=2, show_default=True, help='Number of transition points in the loop chain. 2=A→B→A, 3=A→B→C→A, etc.')
@click.option('--min-segment-duration', type=click.FloatRange(min=1.0), default=4.0, show_default=True, help='Minimum segment duration in seconds.')
@click.option('--max-segment-duration', type=click.FloatRange(min=1.0), default=None, help='Maximum segment duration in seconds.')
def multi_hop(**kwargs):
    """Find multi-hop loop chains with multiple seamless transitions.
    
    Instead of one loop point (start → end → start), finds a chain of
    segments that transition seamlessly into each other.
    
    \b
    Example with n_hops=2: Play intro, then loop A→B→A repeatedly.
    Example with n_hops=3: Play intro, then loop A→B→C→A repeatedly.
    """
    from rich.panel import Panel
    from rich.box import ROUNDED
    
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
            task = progress.add_task("[cyan]Extracting features...", total=None)
            looper = MusicLooper(path)
            from pymusiclooper.analysis import extract_features
            features = extract_features(looper.mlaudio, None, None, False)
            
            progress.update(task, description="[cyan]Finding similar sections...")
            chains = find_multi_hop_loops(
                looper.mlaudio,
                features,
                n_hops=n_hops,
                min_segment_duration=min_seg,
                max_segment_duration=max_seg,
            )
        
        if not chains:
            rich_console.print()
            rich_console.print(Panel(
                "[yellow]No multi-hop chains found[/]\n\n"
                "[dim]Try adjusting parameters:[/]\n"
                "  • Lower [cyan]--min-segment-duration[/]\n"
                "  • Use single-loop mode instead",
                title="[yellow]⚠ No Results[/]",
                border_style="yellow",
            ))
            return
        
        # Display results
        in_samples = "PML_DISPLAY_SAMPLES" in os.environ
        filename = os.path.basename(path)
        
        rich_console.print()
        rich_console.print(Panel(
            f"[bold white]{filename}[/]\n"
            f"[dim]n_hops={n_hops} | min_segment={min_seg}s | chains found: {len(chains)}[/]",
            title="[bold cyan]Multi-Hop Analysis[/]",
            border_style="cyan",
        ))
        
        def format_chain_for_table(chain, idx):
            seg_strs = []
            for start_s, end_s in chain.segment_samples:
                if in_samples:
                    seg_strs.append(f"{start_s}→{end_s}")
                else:
                    seg_strs.append(f"{looper.samples_to_ftime(start_s)}→{looper.samples_to_ftime(end_s)}")
            
            # Score color
            if chain.total_score >= 0.8:
                score_style = "bold green"
            elif chain.total_score >= 0.6:
                score_style = "yellow"
            else:
                score_style = "red"
            
            return seg_strs, [f"{t:.2f}" for t in chain.transitions], score_style
        
        def create_chain_table(chains_to_show, title):
            table = Table(
                title=title,
                box=ROUNDED,
                header_style="bold cyan",
                border_style="dim",
            )
            table.add_column("#", justify="right", style="dim", width=4)
            table.add_column("Segments", style="white", no_wrap=False)
            table.add_column("Transitions", style="dim", width=18)
            table.add_column("Score", justify="right", width=8)
            
            for idx, chain in enumerate(chains_to_show):
                seg_strs, trans_strs, score_style = format_chain_for_table(chain, idx)
                table.add_row(
                    str(idx),
                    " │ ".join(seg_strs),
                    ", ".join(trans_strs),
                    f"[{score_style}]{chain.total_score:.1%}[/]",
                )
            return table
        
        rich_console.print(create_chain_table(chains[:10], f"Top {min(10, len(chains))} Chains"))
        
        # Show best chain details
        best = chains[0]
        best_details = []
        total_duration = 0
        for i, (start_s, end_s) in enumerate(best.segment_samples):
            dur = looper.samples_to_seconds(end_s - start_s)
            total_duration += dur
            if in_samples:
                best_details.append(f"  [cyan]Segment {i+1}:[/] {start_s} → {end_s} [dim]({dur:.1f}s)[/]")
            else:
                best_details.append(f"  [cyan]Segment {i+1}:[/] {looper.samples_to_ftime(start_s)} → {looper.samples_to_ftime(end_s)} [dim]({dur:.1f}s)[/]")
        best_details.append(f"  [dim]↺ Loop back to Segment 1[/]")
        
        rich_console.print()
        rich_console.print(Panel(
            f"[bold green]Score: {best.total_score:.1%}[/] │ "
            f"[dim]Total duration: {total_duration:.1f}s[/]\n\n" +
            "\n".join(best_details),
            title="[bold green]★ Best Chain[/]",
            border_style="green",
        ))
        
        # Interactive preview
        if "PML_INTERACTIVE_MODE" in os.environ:
            rich_console.print()
            rich_console.rule("[bold cyan]Interactive Mode[/]", style="cyan")
            
            # Command hints table
            cmd_table = Table(show_header=False, box=None, padding=(0, 2))
            cmd_table.add_column("Key", style="cyan bold", width=12)
            cmd_table.add_column("Action", style="dim")
            cmd_table.add_row("NUMBER", "Play chain (e.g., 0)")
            cmd_table.add_row("p NUMBER", "Preview with details")
            cmd_table.add_row("more", "Show more chains")
            cmd_table.add_row("q", "Quit")
            rich_console.print(cmd_table)
            rich_console.print()
            
            show_top = 10
            while True:
                try:
                    user_input = rich_console.input("[cyan bold]❯[/] ").strip()
                    
                    if user_input.lower() == 'q':
                        break
                    elif user_input.lower() == 'more':
                        show_top = min(len(chains), show_top + 10)
                        rich_console.print(create_chain_table(chains[:show_top], f"Top {show_top} Chains"))
                        continue
                    
                    preview = user_input.lower().startswith('p')
                    if preview:
                        user_input = user_input[1:].strip()
                    
                    idx = int(user_input)
                    if 0 <= idx < len(chains):
                        chain = chains[idx]
                        
                        if preview:
                            details = []
                            total_dur = 0
                            for i, (start_s, end_s) in enumerate(chain.segment_samples):
                                dur = looper.samples_to_seconds(end_s - start_s)
                                total_dur += dur
                                if in_samples:
                                    details.append(f"  [cyan]Seg {i+1}:[/] {start_s} → {end_s} [dim]({dur:.1f}s)[/]")
                                else:
                                    details.append(f"  [cyan]Seg {i+1}:[/] {looper.samples_to_ftime(start_s)} → {looper.samples_to_ftime(end_s)} [dim]({dur:.1f}s)[/]")
                            details.append(f"  [dim]↺ Loop[/]")
                            
                            rich_console.print(Panel(
                                f"[bold]Score:[/] {chain.total_score:.1%} │ "
                                f"[dim]Duration: {total_dur:.1f}s[/]\n"
                                f"[bold]Transitions:[/] {', '.join([f'{t:.2f}' for t in chain.transitions])}\n\n" +
                                "\n".join(details),
                                title=f"[bold]Chain #{idx}[/]",
                                border_style="blue",
                            ))
                        
                        rich_console.print(f"[bold green]▶ Playing chain #{idx}[/] [dim](Ctrl+C to stop)[/]")
                        looper.play_multi_hop(chain.segment_samples)
                    else:
                        rich_console.print(f"[red]✗[/] Index must be 0-{len(chains)-1}")
                except ValueError:
                    rich_console.print("[red]✗[/] Invalid input. Type a number, 'p0', 'more', or 'q'")
                except KeyboardInterrupt:
                    rich_console.print("\n[dim]Stopped[/]")
                    
    except (AudioLoadError, LoopNotFoundError, Exception) as e:
        print_exception(e)


@cli_main.command()
@click.option('--path', type=click.Path(exists=True), required=True, help='Path to the audio file.')
@click.option('--n-hops', type=click.IntRange(min=2, max=10), default=2, show_default=True, help='Number of transition points.')
@click.option('--min-segment-duration', type=click.FloatRange(min=1.0), default=4.0, show_default=True, help='Minimum segment duration in seconds.')
@click.option('--max-segment-duration', type=click.FloatRange(min=1.0), default=None, help='Maximum segment duration in seconds.')
@click.option('--output-dir', '-o', type=click.Path(exists=False, writable=True, file_okay=False), help="Output directory for the export.")
@click.option('--top', type=int, default=1, show_default=True, help='Export top N chains.')
@click.option('--format', '-f', 'fmt', type=click.Choice(['txt', 'json']), default='txt', show_default=True, help='Output format.')
def export_multi_hop(**kwargs):
    """Export multi-hop loop chain points to a file.
    
    \b
    Output formats:
      txt  - Human-readable with comments
      json - Machine-readable structured data
    """
    import json as json_lib
    from rich.panel import Panel
    from rich.box import ROUNDED
    
    try:
        path = kwargs["path"]
        n_hops = kwargs["n_hops"]
        min_seg = kwargs["min_segment_duration"]
        max_seg = kwargs["max_segment_duration"]
        output_dir = kwargs.get("output_dir") or os.path.dirname(path) or "."
        top_n = kwargs["top"]
        fmt = kwargs["fmt"]
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=rich_console,
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Extracting features...", total=None)
            looper = MusicLooper(path)
            from pymusiclooper.analysis import extract_features
            features = extract_features(looper.mlaudio, None, None, False)
            
            progress.update(task, description="[cyan]Finding similar sections...")
            chains = find_multi_hop_loops(
                looper.mlaudio,
                features,
                n_hops=n_hops,
                min_segment_duration=min_seg,
                max_segment_duration=max_seg,
            )
        
        if not chains:
            rich_console.print()
            rich_console.print(Panel(
                "[yellow]No multi-hop chains found[/]",
                title="[yellow]⚠ No Results[/]",
                border_style="yellow",
            ))
            return
        
        # Export to file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.splitext(os.path.basename(path))[0]
        ext = 'json' if fmt == 'json' else 'txt'
        out_path = os.path.join(output_dir, f"{filename}_multihop.{ext}")
        
        if fmt == 'json':
            export_data = {
                "source": os.path.basename(path),
                "n_hops": n_hops,
                "sample_rate": int(looper.mlaudio.rate),
                "chains": []
            }
            for idx, chain in enumerate(chains[:top_n]):
                chain_data = {
                    "index": idx,
                    "score": round(float(chain.total_score), 4),
                    "transitions": [round(float(t), 4) for t in chain.transitions],
                    "segments": [
                        {
                            "start_sample": int(start_s),
                            "end_sample": int(end_s),
                            "start_time": looper.samples_to_ftime(start_s),
                            "end_time": looper.samples_to_ftime(end_s),
                            "duration_s": round(float(looper.samples_to_seconds(end_s - start_s)), 3)
                        }
                        for start_s, end_s in chain.segment_samples
                    ]
                }
                export_data["chains"].append(chain_data)
            
            with open(out_path, "w") as f:
                json_lib.dump(export_data, f, indent=2)
        else:
            with open(out_path, "w") as f:
                f.write(f"# PyMusicLooper Multi-Hop Export\n")
                f.write(f"# Source: {os.path.basename(path)}\n")
                f.write(f"# Parameters: n_hops={n_hops}, min_segment={min_seg}s\n")
                f.write(f"# Sample Rate: {looper.mlaudio.rate}\n")
                f.write(f"#\n")
                f.write(f"# Format: INDEX SCORE START1 END1 START2 END2 ...\n")
                f.write(f"# (All positions in samples)\n\n")
                
                for idx, chain in enumerate(chains[:top_n]):
                    parts = [str(idx), f"{chain.total_score:.4f}"]
                    for start_s, end_s in chain.segment_samples:
                        parts.extend([str(start_s), str(end_s)])
                    f.write(" ".join(parts) + "\n")
        
        # Display results
        rich_console.print()
        
        best = chains[0]
        total_dur = sum(looper.samples_to_seconds(e - s) for s, e in best.segment_samples)
        
        seg_lines = []
        for i, (start_s, end_s) in enumerate(best.segment_samples):
            dur = looper.samples_to_seconds(end_s - start_s)
            seg_lines.append(
                f"  [cyan]Segment {i+1}:[/] {looper.samples_to_ftime(start_s)} → {looper.samples_to_ftime(end_s)} "
                f"[dim]({dur:.1f}s)[/]"
            )
        
        rich_console.print(Panel(
            f"[bold green]✓ Exported {min(top_n, len(chains))} chain(s)[/]\n"
            f"[dim]Output:[/] {out_path}\n\n"
            f"[bold]Best Chain (#{0}):[/]\n"
            f"  Score: [green]{best.total_score:.1%}[/] │ Duration: {total_dur:.1f}s\n\n" +
            "\n".join(seg_lines),
            title="[bold cyan]Export Complete[/]",
            border_style="green",
        ))
            
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
        
        from rich.panel import Panel
        from rich.box import ROUNDED
        
        in_samples = "PML_DISPLAY_SAMPLES" in os.environ
        filename = os.path.basename(path)
        
        # Header panel
        rich_console.print()
        rich_console.print(Panel(
            f"[bold white]{filename}[/]\n"
            f"[dim]Segment:[/] [cyan]{start_sec:.2f}s[/] → [cyan]{end_sec:.2f}s[/] "
            f"[dim]({end_sec - start_sec:.1f}s)[/]\n"
            f"[dim]Mode:[/] [green]{mode}[/] │ [dim]Found:[/] {len(results)} loops",
            title="[bold cyan]Segment Loop Analysis[/]",
            border_style="cyan",
        ))
        
        # Results table
        table = Table(
            title=f"Top {len(results)} Loop Points",
            box=ROUNDED,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Start", style="green", width=10)
        table.add_column("End", style="green", width=10)
        table.add_column("Duration", style="dim", width=9)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Quality", width=5)
        
        for idx, result in enumerate(results):
            if in_samples:
                start_str = str(result.loop_start)
                end_str = str(result.loop_end)
            else:
                start_str = looper.samples_to_ftime(result.loop_start)
                end_str = looper.samples_to_ftime(result.loop_end)
            
            duration = looper.samples_to_seconds(result.loop_end - result.loop_start)
            
            # Score styling
            if result.score >= 0.8:
                score_style = "bold green"
                quality = "[green]★★★[/]"
            elif result.score >= 0.6:
                score_style = "yellow"
                quality = "[yellow]★★[/]"
            else:
                score_style = "red"
                quality = "[red]★[/]"
            
            table.add_row(
                str(idx),
                start_str,
                end_str,
                f"{duration:.2f}s",
                f"[{score_style}]{result.score:.1%}[/]",
                result.boundary_type,
                quality,
            )
        
        rich_console.print(table)
        
        # Best result details panel
        best = results[0]
        best_score_style = "green" if best.score >= 0.7 else ("yellow" if best.score >= 0.5 else "red")
        rich_console.print(Panel(
            f"[{best_score_style}]{best.explanation}[/]\n\n"
            f"[bold]Metrics:[/]\n"
            f"  Rhythm:     [{best_score_style if best.rhythm_alignment >= 0.7 else 'dim'}]{best.rhythm_alignment:.1%}[/]\n"
            f"  Harmony:    [{best_score_style if best.harmonic_match >= 0.7 else 'dim'}]{best.harmonic_match:.1%}[/]\n"
            f"  Transition: [{best_score_style if best.transition_quality >= 0.7 else 'dim'}]{best.transition_quality:.1%}[/]",
            title="[bold green]★ Best Loop[/]",
            border_style="green",
        ))
        
        # Interactive mode
        if "PML_INTERACTIVE_MODE" in os.environ:
            rich_console.print()
            rich_console.rule("[bold cyan]Interactive Mode[/]", style="cyan")
            
            cmd_table = Table(show_header=False, box=None, padding=(0, 2))
            cmd_table.add_column("Key", style="cyan bold", width=10)
            cmd_table.add_column("Action", style="dim")
            cmd_table.add_row("NUMBER", "Play loop (e.g., 0)")
            cmd_table.add_row("q", "Quit")
            rich_console.print(cmd_table)
            rich_console.print()
            
            while True:
                try:
                    user_input = rich_console.input("[cyan bold]❯[/] ").strip()
                    
                    if user_input.lower() == 'q':
                        break
                    
                    idx = int(user_input)
                    if 0 <= idx < len(results):
                        result = results[idx]
                        dur = looper.samples_to_seconds(result.loop_end - result.loop_start)
                        rich_console.print(Panel(
                            f"[bold]Loop #{idx}[/]\n"
                            f"[green]{looper.samples_to_ftime(result.loop_start)}[/] → "
                            f"[green]{looper.samples_to_ftime(result.loop_end)}[/] "
                            f"[dim]({dur:.1f}s)[/]",
                            title="[bold green]▶ Playing[/]",
                            subtitle="[dim]Ctrl+C to stop[/]",
                            border_style="green",
                        ))
                        looper.play_looping(result.loop_start, result.loop_end)
                    else:
                        rich_console.print(f"[red]✗[/] Index must be 0-{len(results)-1}")
                        
                except ValueError:
                    rich_console.print("[red]✗[/] Enter a number or 'q'")
                except KeyboardInterrupt:
                    rich_console.print("\n[dim]Stopped[/]")
        
    except (AudioLoadError, LoopNotFoundError, Exception) as e:
        print_exception(e)


@cli_main.command()
@click.option('--path', type=click.Path(exists=True), required=True, help='Path to the audio file.')
@click.option('--top', type=int, default=5, show_default=True, help='Analyze top N loops.')
@click.option('--criterion', type=click.Choice(['overall', 'melodicity', 'rhythm', 'seamless']), default='overall', show_default=True, help='Ranking criterion.')
def quality(**kwargs):
    """Analyze loop quality with detailed metrics.
    
    Provides comprehensive quality analysis including:
    
    \b
    • Melodicity and melodic continuity
    • Harmonic coherence and consonance
    • Rhythmic integrity and phase alignment
    • Perceptual imperceptibility
    • Crossfade quality
    
    Example:
    
    \b
        pymusiclooper quality --path song.mp3 --criterion melodicity
    """
    from rich.panel import Panel
    from rich.box import ROUNDED
    
    try:
        path = kwargs["path"]
        top_n = kwargs["top"]
        criterion = kwargs["criterion"]
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=rich_console,
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Analyzing...", total=None)
            
            looper = MusicLooper(path)
            
            progress.update(task, description="[cyan]Finding loops...")
            loops = looper.find_loop_pairs()
            
            progress.update(task, description="[cyan]Computing quality metrics...")
            from pymusiclooper.analysis import (
                extract_features, rank_loops_by_quality, compute_quality_metrics
            )
            features = extract_features(looper.mlaudio, None, None, False)
            
            ranked = rank_loops_by_quality(features, loops[:top_n * 2], criterion)
        
        # Header panel
        filename = os.path.basename(path)
        rich_console.print()
        rich_console.print(Panel(
            f"[bold white]{filename}[/]\n"
            f"[dim]Criterion:[/] [cyan]{criterion}[/] │ "
            f"[dim]Loops analyzed:[/] {len(ranked)}",
            title="[bold cyan]Quality Analysis[/]",
            border_style="cyan",
        ))
        
        # Results table
        table = Table(
            title=f"Top {min(top_n, len(ranked))} Loops by {criterion.title()}",
            box=ROUNDED,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Start", style="green", width=10)
        table.add_column("End", style="green", width=10)
        table.add_column("Overall", justify="right", width=8)
        table.add_column("Melodic", justify="right", width=8)
        table.add_column("Rhythm", justify="right", width=8)
        table.add_column("Harmonic", justify="right", width=8)
        table.add_column("Seamless", justify="right", width=8)
        
        in_samples = "PML_DISPLAY_SAMPLES" in os.environ
        
        def score_fmt(val):
            """Format score with color."""
            if val >= 0.8:
                return f"[bold green]{val:.0%}[/]"
            elif val >= 0.6:
                return f"[yellow]{val:.0%}[/]"
            else:
                return f"[red]{val:.0%}[/]"
        
        for idx, (loop, metrics) in enumerate(ranked[:top_n]):
            if in_samples:
                start_str = str(loop.loop_start)
                end_str = str(loop.loop_end)
            else:
                start_str = looper.samples_to_ftime(loop.loop_start)
                end_str = looper.samples_to_ftime(loop.loop_end)
            
            table.add_row(
                str(idx),
                start_str,
                end_str,
                score_fmt(metrics.overall_score),
                score_fmt(metrics.melodicity),
                score_fmt(metrics.rhythm_integrity),
                score_fmt(metrics.harmonic_coherence),
                score_fmt(metrics.imperceptibility),
            )
        
        rich_console.print(table)
        
        # Detailed view of best loop
        if ranked:
            best_loop, best_metrics = ranked[0]
            
            # Grade calculation
            score = best_metrics.overall_score
            if score >= 0.9:
                grade, grade_style = "A+", "bold green"
            elif score >= 0.85:
                grade, grade_style = "A", "bold green"
            elif score >= 0.8:
                grade, grade_style = "A-", "green"
            elif score >= 0.75:
                grade, grade_style = "B+", "green"
            elif score >= 0.7:
                grade, grade_style = "B", "yellow"
            elif score >= 0.65:
                grade, grade_style = "B-", "yellow"
            elif score >= 0.6:
                grade, grade_style = "C+", "yellow"
            else:
                grade, grade_style = "C", "red"
            
            rich_console.print(Panel(
                f"[bold]Overall Quality:[/] [{grade_style}]{score:.1%}[/] ([{grade_style}]{grade}[/])\n\n"
                f"[bold cyan]Melodic[/]\n"
                f"  Melodicity:  {score_fmt(best_metrics.melodicity)} │ Continuity: {score_fmt(best_metrics.melodic_continuity)}\n\n"
                f"[bold cyan]Harmonic[/]\n"
                f"  Coherence:   {score_fmt(best_metrics.harmonic_coherence)} │ Consonance: {score_fmt(best_metrics.consonance)}\n\n"
                f"[bold cyan]Rhythmic[/]\n"
                f"  Integrity:   {score_fmt(best_metrics.rhythm_integrity)} │ Phase:      {score_fmt(best_metrics.phase_alignment)}\n\n"
                f"[bold cyan]Timbral[/]\n"
                f"  Match:       {score_fmt(best_metrics.timbral_match)} │ Stability:  {score_fmt(best_metrics.spectral_stability)}\n\n"
                f"[bold cyan]Technical[/]\n"
                f"  Crossfade:   {score_fmt(best_metrics.crossfade_quality)} │ Transient:  {score_fmt(best_metrics.transient_avoidance)}\n\n"
                f"[dim]Confidence: {best_metrics.confidence:.0%}[/]",
                title="[bold green]★ Best Loop Details[/]",
                border_style="green",
            ))
            
            # Interactive mode
            if "PML_INTERACTIVE_MODE" in os.environ:
                rich_console.print()
                rich_console.rule("[bold cyan]Interactive Mode[/]", style="cyan")
                
                cmd_table = Table(show_header=False, box=None, padding=(0, 2))
                cmd_table.add_column("Key", style="cyan bold", width=10)
                cmd_table.add_column("Action", style="dim")
                cmd_table.add_row("NUMBER", "Play loop (e.g., 0)")
                cmd_table.add_row("q", "Quit")
                rich_console.print(cmd_table)
                rich_console.print()
                
                while True:
                    try:
                        user_input = rich_console.input("[cyan bold]❯[/] ").strip()
                        if user_input.lower() == 'q':
                            break
                        idx = int(user_input)
                        if 0 <= idx < len(ranked):
                            loop, metrics = ranked[idx]
                            dur = looper.samples_to_seconds(loop.loop_end - loop.loop_start)
                            rich_console.print(Panel(
                                f"[bold]Loop #{idx}[/]\n"
                                f"[green]{looper.samples_to_ftime(loop.loop_start)}[/] → "
                                f"[green]{looper.samples_to_ftime(loop.loop_end)}[/] "
                                f"[dim]({dur:.1f}s)[/]\n"
                                f"Quality: [green]{metrics.overall_score:.0%}[/]",
                                title="[bold green]▶ Playing[/]",
                                subtitle="[dim]Ctrl+C to stop[/]",
                                border_style="green",
                            ))
                            looper.play_looping(loop.loop_start, loop.loop_end)
                        else:
                            rich_console.print(f"[red]✗[/] Index must be 0-{len(ranked)-1}")
                    except ValueError:
                        rich_console.print("[red]✗[/] Enter a number or 'q'")
                    except KeyboardInterrupt:
                        rich_console.print("\n[dim]Stopped[/]")
    
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
