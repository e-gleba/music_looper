"""
Console utilities and Rich formatting for PyMusicLooper.

Provides professional CLI output with Rich library:
- Styled tables and panels
- Progress indicators
- Status messages
- Interactive prompts
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.box import ROUNDED, HEAVY, DOUBLE
from rich import box

# Module-level rich console instance
rich_console = Console()

# ============================================================================
# STYLES
# ============================================================================

STYLE_SUCCESS = Style(color="green", bold=True)
STYLE_ERROR = Style(color="red", bold=True)
STYLE_WARNING = Style(color="yellow")
STYLE_INFO = Style(color="cyan")
STYLE_DIM = Style(dim=True)
STYLE_HEADER = Style(color="bright_white", bold=True)
STYLE_SCORE_HIGH = Style(color="green", bold=True)
STYLE_SCORE_MED = Style(color="yellow")
STYLE_SCORE_LOW = Style(color="red")


# ============================================================================
# UI COMPONENTS
# ============================================================================

def print_header(title: str, subtitle: str = None):
    """Print a styled header."""
    header_text = Text(title, style=STYLE_HEADER)
    if subtitle:
        header_text.append(f"\n{subtitle}", style=STYLE_DIM)
    
    panel = Panel(
        header_text,
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 2),
    )
    rich_console.print(panel)


def print_status(message: str, status: str = "info"):
    """Print a styled status message."""
    icons = {
        "success": ("✓", STYLE_SUCCESS),
        "error": ("✗", STYLE_ERROR),
        "warning": ("⚠", STYLE_WARNING),
        "info": ("•", STYLE_INFO),
    }
    icon, style = icons.get(status, ("•", STYLE_INFO))
    rich_console.print(f"[{style.color}]{icon}[/] {message}")


def print_tip(message: str):
    """Print a helpful tip."""
    rich_console.print(f"[dim]💡 {message}[/]")


def print_command_hint(commands: dict[str, str]):
    """Print command hints in a professional format."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        collapse_padding=True,
    )
    table.add_column("Key", style="cyan bold", no_wrap=True)
    table.add_column("Action", style="dim")
    
    for key, action in commands.items():
        table.add_row(key, action)
    
    rich_console.print(table)


def score_to_style(score: float) -> Style:
    """Get appropriate style for a score value."""
    if score >= 0.75:
        return STYLE_SCORE_HIGH
    elif score >= 0.5:
        return STYLE_SCORE_MED
    else:
        return STYLE_SCORE_LOW


def format_score(score: float, width: int = 6) -> Text:
    """Format a score with appropriate coloring."""
    text = f"{score:.1%}".rjust(width)
    style = score_to_style(score)
    return Text(text, style=style)


def format_time(time_str: str) -> Text:
    """Format a timestamp."""
    return Text(time_str, style="green")


def format_duration(duration: float) -> str:
    """Format duration in seconds to human readable."""
    if duration < 60:
        return f"{duration:.1f}s"
    else:
        mins = int(duration // 60)
        secs = duration % 60
        return f"{mins}m {secs:.1f}s"


def create_results_table(
    title: str,
    columns: list[tuple[str, str, str]],  # (name, style, justify)
) -> Table:
    """Create a styled results table."""
    table = Table(
        title=title,
        box=ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        row_styles=["", "dim"],
    )
    
    for name, style, justify in columns:
        table.add_column(name, style=style, justify=justify)
    
    return table


def print_interactive_prompt():
    """Print the interactive mode prompt."""
    rich_console.print()
    rich_console.rule("[bold cyan]Interactive Mode", style="cyan")


def print_loop_info(
    loop_start: str,
    loop_end: str,
    score: float,
    duration: float = None,
):
    """Print loop point information in a styled panel."""
    content = Text()
    content.append("Start: ", style="dim")
    content.append(loop_start, style="green bold")
    content.append(" → ", style="dim")
    content.append("End: ", style="dim")
    content.append(loop_end, style="green bold")
    content.append("\n")
    content.append("Score: ", style="dim")
    content.append(f"{score:.1%}", style=score_to_style(score))
    
    if duration:
        content.append(" │ ", style="dim")
        content.append("Duration: ", style="dim")
        content.append(format_duration(duration), style="cyan")
    
    panel = Panel(content, box=ROUNDED, border_style="green")
    rich_console.print(panel)


# ============================================================================
# CLI HELP STYLING
# ============================================================================

# Creating groups for CLI --help styling
_basic_options = ["--path", "--url"]
_loop_options = [
    "--min-duration-multiplier",
    "--min-loop-duration",
    "--max-loop-duration",
    "--approx-loop-position",
    "--brute-force",
    "--disable-pruning",
]
_export_options = ["--output-dir", "--format"]
_batch_options = ["--recursive", "--flatten"]


def _option_groups(additional_basic_options=None):
    if additional_basic_options is not None:
        combined_basic_options = _basic_options + additional_basic_options
    else:
        combined_basic_options = _basic_options
    return [
        {
            "name": "Basic options",
            "options": combined_basic_options,
        },
        {
            "name": "Advanced loop options",
            "options": _loop_options,
        },
        {
            "name": "Export options",
            "options": _export_options,
        },
        {
            "name": "Batch options",
            "options": _batch_options,
        },
    ]


_common_option_groups = _option_groups()
_OPTION_GROUPS = {
    "pymusiclooper play": _common_option_groups,
    "pymusiclooper split-audio": _common_option_groups,
    "pymusiclooper tag": _option_groups(["--tag-names", "--tag-offset"]),
    "pymusiclooper export-points": _option_groups(["--export-to", "--alt-export-top", "--fmt"]),
    "pymusiclooper extend": _option_groups(["--extended-length", "--fade-length", "--disable-fade-out"]),
}

_COMMAND_GROUPS = {
    "pymusiclooper": [
        {
            "name": "Play Commands",
            "commands": [
                "play",
                "play-tagged"
            ],
        },
        {
            "name": "Export Commands",
            "commands": [
                "export-points",
                "split-audio",
                "tag",
                "extend",
            ],
        },
        {
            "name": "Analysis Commands",
            "commands": [
                "quality",
                "segment-loop",
                "multi-hop",
                "export-multi-hop",
            ],
        }
    ]
}
