"""
src/cli/main.py
===============
TrustScript CLI — Main Group and Shared Utilities

Owns:
    - The `trustscript` Click group definition.
    - Logging configuration shared across all subcommands.
    - The Rich console instance used for all terminal output.

Does not own:
    - Any subcommand. Those live in their own files (diarize.py,
      overlap.py, etc.) and are registered in __init__.py.

Contributors
------------
    Add shared CLI utilities here (e.g. a shared progress bar factory,
    a common error formatter). Do not add subcommand logic here.
"""

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Shared Rich console
#
# Import this from other cli modules when you need to print to the terminal.
# Using a single shared console instance ensures Rich's output doesn't
# interleave incorrectly during batch processing.
# ---------------------------------------------------------------------------

console = Console()


def configure_logging(verbose: bool, log_file: Path | None = None) -> None:
    """
    Configure the root logger for a pipeline run.

    Normal mode:  INFO level, Rich-formatted output to stderr.
    Verbose mode: DEBUG level, includes file/line info and per-segment detail.
    Log file:     Additionally writes plain-text logs to disk when provided.

    Called once at the start of each subcommand before any pipeline work.

    Parameters
    ----------
    verbose : bool
        True to enable DEBUG-level logging.
    log_file : Path | None
        Optional path to write a plain-text log file alongside stdout.
        Parent directories are created if they do not exist.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=verbose,  # Show file:line in verbose mode only.
        )
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # Override any handlers already set before CLI entry.
    )


@click.group()
@click.version_option(package_name="trustscript", prog_name="trustscript")
def trustscript() -> None:
    """
    OpenBWC TrustScript: Forensic-grade speaker diarization for body-worn camera footage.

    Processes BWC video and audio to produce a structured speaker timeline
    with calibrated confidence scores, chain-of-custody metadata, and
    explicit uncertainty quantification at every stage.

    Run `trustscript COMMAND --help` for help on a specific command.

    \b
    Pipeline phases:
        diarize     Phase 1 — Speaker diarization and vault construction
        overlap     Phase 2 — Overlap characterization and triage    [coming]
        transcribe  Phase 3 — Dual-engine ASR (Whisper + Kaldi)      [coming]
        fuse        Phase 4 — Multi-source confidence fusion         [coming]
        package     Phase 5 — TrustScript output assembly            [coming]
        run         Full pipeline (all phases in sequence)           [coming]
    """