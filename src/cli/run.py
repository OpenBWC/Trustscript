# ==============================================================================
# src/cli/run.py
# ==============================================================================
"""
TrustScript CLI — Full Pipeline Runner  [STUB]
"""

import click
from .main import configure_logging


@click.command(name="run")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True))
@click.option("--output", "-o", default=None,
              type=click.Path(file_okay=False, writable=True))
@click.option("--no-vault", is_flag=True, default=False)
@click.option("--no-timeline", is_flag=True, default=False)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--chunk-size", default=300, show_default=True,
              type=click.IntRange(min=30))
@click.option("--token", default=None, envvar="HF_TOKEN")
@click.option("--models-dir", default=None,
              type=click.Path(exists=True, file_okay=False))
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option("--log-file", default=None,
              type=click.Path(dir_okay=False, writable=True))
def run_pipeline(
    input_path, output, no_vault, no_timeline, dry_run,
    chunk_size, token, models_dir, verbose, log_file,
) -> None:
    """
    Run the full TrustScript pipeline (all phases in sequence).  [NOT IMPLEMENTED]

    Executes Phase 1 through Phase 5 for each input file.
    Equivalent to: diarize → overlap → transcribe → fuse → package
    with shared state passed between phases.

    As individual phases are implemented, this command will chain them
    automatically. Run phases individually in the meantime.
    """
    from pathlib import Path
    configure_logging(verbose, Path(log_file) if log_file else None)
    raise NotImplementedError(
        "Full pipeline runner is not yet implemented. "
        "Run phases individually using their subcommands."
    )