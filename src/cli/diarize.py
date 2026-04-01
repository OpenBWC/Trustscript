"""
src/cli/diarize.py
==================
TrustScript CLI — Phase 1: Speaker Diarization Engine

Owns:
    - The `trustscript diarize` Click command and all its flags.
    - The `_run_diarize_pipeline()` private orchestrator that calls
      each Stage 1 stage function in sequence.

Stage status
------------
Each stage block in _run_diarize_pipeline() is clearly labeled with
its implementation status. To implement a stage:
    1. Remove the `raise NotImplementedError(...)` line.
    2. Import and call the stage function.
    3. Pass the result to the next stage.

Contributors
------------
    Add new --diarize-specific flags to the `diarize` command below.
    Flags shared across multiple subcommands belong in main.py.
    Do not add Phase 2+ logic to this file.
"""

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .main import configure_logging, console


# ---------------------------------------------------------------------------
# Command definition
# ---------------------------------------------------------------------------


@click.command()
# --- Input ---
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help=(
        "Path to a single BWC video/audio file, or a directory for batch "
        "processing. Supported: .mp4 .mov .avi .mkv .wav .mp3 .m4a .flac"
    ),
)
# --- Output ---
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(file_okay=False, writable=True),
    help=(
        "Directory to write output files. "
        "Defaults to data/ in the repo root if not specified."
    ),
)
@click.option(
    "--no-vault",
    is_flag=True,
    default=False,
    help=(
        "Suppress {incident_id}_vault.json. "
        "Use on storage-constrained hardware. "
        "Note: vault.json is consumed by Phase 4 Fusion — "
        "suppressing it will prevent future fusion runs."
    ),
)
@click.option(
    "--no-timeline",
    is_flag=True,
    default=False,
    help=(
        "Suppress {incident_id}_timeline.json. "
        "Note: timeline.json is consumed by Phase 3 ASR — "
        "suppressing it will prevent future transcription runs."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Execute all pipeline stages but write no output files. "
        "Useful for validating a deployment or testing a new footage "
        "format without producing artifacts."
    ),
)
# --- Identity ---
@click.option(
    "--incident-id",
    default=None,
    help=(
        "Override the incident ID used to name output files. "
        "Defaults to the input filename stem (e.g. 'incident_001' from "
        "'incident_001.mp4'). Useful when camera-generated filenames "
        "are not human-readable (e.g. 'CAM_20241124_083212.mp4')."
    ),
)
# --- Diarization tuning ---
@click.option(
    "--chunk-size",
    default=300,
    show_default=True,
    type=click.IntRange(min=30),
    help=(
        "Diarization window size in seconds. Audio is processed in chunks "
        "of this length with a fixed 10-second overlap at boundaries. "
        "Smaller values use less RAM but create more vault matching "
        "operations and boundary seams. "
        "Recommended: 300s (5 min) for production BWC footage."
    ),
)
# --- Model / auth ---
@click.option(
    "--token",
    default=None,
    envvar="HF_TOKEN",
    help=(
        "HuggingFace access token for downloading gated pyannote models. "
        "Resolution order: (1) this flag, (2) HF_TOKEN in .env, "
        "(3) HF_TOKEN environment variable. "
        "One-time setup — token is cached after first model download."
    ),
)
@click.option(
    "--models-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Path to a local directory containing pre-downloaded pyannote "
        "model weights. Use for offline or air-gapped deployments where "
        "the processing machine cannot reach HuggingFace. "
        "See README.md — Offline Deployment for setup instructions."
    ),
)
# --- Dev / debug ---
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help=(
        "Enable DEBUG-level logging. Prints per-segment reid scores, "
        "vault gate decisions, and ffprobe output during processing."
    ),
)
@click.option(
    "--log-file",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help=(
        "Write a plain-text log file in addition to stdout. "
        "Useful for batch runs where a permanent record is needed. "
        "Example: --log-file logs/run_2024_11_24.log"
    ),
)
def diarize(
    input_path: str,
    output: str | None,
    no_vault: bool,
    no_timeline: bool,
    dry_run: bool,
    incident_id: str | None,
    chunk_size: int,
    token: str | None,
    models_dir: str | None,
    verbose: bool,
    log_file: str | None,
) -> None:
    """
    Phase 1 — Speaker Diarization Engine.

    Processes BWC footage through a 7-stage pipeline to produce a
    structured speaker timeline with calibrated confidence scores and
    chain-of-custody metadata.

    \b
    Output files (all written by default):
        {incident_id}_phase1.json    Combined metadata, speakers,
                                     timeline, and quality summary.
        {incident_id}_vault.json     Full speaker embedding history.
                                     Consumed by Phase 4 Fusion.
        {incident_id}_timeline.json  Speaker segments.
                                     Consumed by Phase 3 ASR.

    \b
    Examples:
        trustscript diarize --input footage/incident_001.mp4
        trustscript diarize --input footage/ --output results/ --verbose
        trustscript diarize --input footage/ --chunk-size 180 --dry-run
        trustscript diarize --input footage/ --models-dir models/ --no-vault
    """
    load_dotenv()
    configure_logging(verbose, Path(log_file) if log_file else None)
    log = logging.getLogger(__name__)

    input_path_obj = Path(input_path)
    output_path_obj = Path(output) if output else Path("data")
    is_batch = input_path_obj.is_dir()

    if dry_run:
        console.print("[yellow]Dry run — no output files will be written.[/yellow]")

    # Collect files to process.
    if is_batch:
        from src.utils.audio import ALL_SUPPORTED_FORMATS
        files = sorted(
            f for f in input_path_obj.iterdir()
            if f.is_file() and f.suffix.lower() in ALL_SUPPORTED_FORMATS
        )
        if not files:
            console.print(
                f"[red]No supported files found in: {input_path_obj}[/red]"
            )
            sys.exit(1)
        console.print(
            f"Batch mode — found [bold]{len(files)}[/bold] file(s) to process."
        )
    else:
        files = [input_path_obj]

    # Process each file. Batch mode logs failures and continues;
    # single-file mode re-raises so the full traceback is visible.
    results = []
    failed = []

    for file in files:
        file_incident_id = (
            incident_id if (incident_id and not is_batch)
            else file.stem
        )

        console.rule(f"[bold]{file.name}[/bold]  →  incident: {file_incident_id}")

        try:
            _run_diarize_pipeline(
                file=file,
                incident_id=file_incident_id,
                output_dir=output_path_obj,
                no_vault=no_vault,
                no_timeline=no_timeline,
                dry_run=dry_run,
                chunk_size=chunk_size,
                token=token,
                models_dir=Path(models_dir) if models_dir else None,
                verbose=verbose,
            )
            results.append(file)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            sys.exit(130)

        except Exception as e:
            log.error("Failed to process %s: %s", file.name, e)
            failed.append((file, e))
            if not is_batch:
                raise

    # Batch summary.
    if is_batch:
        console.rule("[bold]Batch Complete[/bold]")
        console.print(f"  Processed: [green]{len(results)}[/green] / {len(files)}")
        if failed:
            console.print(f"  Failed:    [red]{len(failed)}[/red]")
            for f, err in failed:
                console.print(f"    [red]✗[/red] {f.name}: {err}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline orchestrator — private
# ---------------------------------------------------------------------------


def _run_diarize_pipeline(
    file: Path,
    incident_id: str,
    output_dir: Path,
    no_vault: bool,
    no_timeline: bool,
    dry_run: bool,
    chunk_size: int,
    token: str | None,
    models_dir: Path | None,
    verbose: bool,
) -> None:
    """
    Execute the Phase 1 diarization pipeline for a single file.

    Calls each stage in sequence. Stages not yet implemented raise
    NotImplementedError with a message indicating their spec location.
    As stages are built, remove the stub and wire in the real call.

    Parameters
    ----------
    file : Path
        Resolved absolute path to the input file.
    incident_id : str
        Identifier used to name all output files for this run.
    output_dir : Path
        Directory to write output files.
    no_vault : bool
        Suppress writing the vault JSON if True.
    no_timeline : bool
        Suppress writing the timeline JSON if True.
    dry_run : bool
        Skip all file writes if True.
    chunk_size : int
        Diarization window size in seconds.
    token : str | None
        HuggingFace access token.
    models_dir : Path | None
        Local model weights directory for offline deployment.
    verbose : bool
        DEBUG logging active if True.
    """
    log = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 — Input Handling & Format Detection
    # Status:  IMPLEMENTED
    # File:    src/utils/audio/stage1.py
    # ------------------------------------------------------------------
    log.info("Stage 1 — Input Handling & Format Detection")

    from src.utils.audio import AudioIngestionError, FFprobeError, run_stage1

    try:
        stage1_result = run_stage1(file)
    except (AudioIngestionError, FFprobeError) as e:
        raise click.ClickException(str(e)) from e

    if verbose:
        console.print(
            f"  [dim]SHA256:[/dim]      {stage1_result.sha256[:16]}...\n"
            f"  [dim]passthrough:[/dim] {stage1_result.passthrough.can_skip_normalization}\n"
            f"  [dim]video input:[/dim] {stage1_result.is_video_input}"
        )

    # ------------------------------------------------------------------
    # Stage 2 — Audio Quality Profiling
    # Status:  NOT IMPLEMENTED
    # File:    src/utils/audio/profiling.py  (to be created)
    # ------------------------------------------------------------------
    log.info("Stage 2 — Audio Quality Profiling")
    raise NotImplementedError(
        "Stage 2 (Audio Quality Profiling) is not yet implemented. "
        "See Phase 1 spec Stage 2 and src/utils/audio/profiling.py."
    )

    # ------------------------------------------------------------------
    # Stage 3 — Normalization
    # Status:  NOT IMPLEMENTED
    # File:    src/utils/audio/normalization.py  (to be created)
    # Note:    Skip this stage when stage1_result.passthrough is True.
    # ------------------------------------------------------------------
    log.info("Stage 3 — Normalization")
    raise NotImplementedError(
        "Stage 3 (Normalization) is not yet implemented. "
        "See Phase 1 spec Stage 3 and src/utils/audio/normalization.py."
    )

    # ------------------------------------------------------------------
    # Stage 4 — Model Loading
    # Status:  NOT IMPLEMENTED
    # File:    src/diarization/models.py
    # ------------------------------------------------------------------
    log.info("Stage 4 — Model Loading")
    raise NotImplementedError(
        "Stage 4 (Model Loading) is not yet implemented. "
        "See Phase 1 spec Stage 4 and src/diarization/models.py."
    )

    # ------------------------------------------------------------------
    # Stage 5 — Windowed Diarization + Vault Matching
    # Status:  NOT IMPLEMENTED
    # Files:   src/diarization/engine.py, src/diarization/vault.py
    # ------------------------------------------------------------------
    log.info("Stage 5 — Windowed Diarization + Vault Matching")
    raise NotImplementedError(
        "Stage 5 (Diarization + Vault Matching) is not yet implemented. "
        "See Phase 1 spec Stage 5 and src/diarization/engine.py."
    )

    # ------------------------------------------------------------------
    # Stage 6 — Retroactive Re-scoring
    # Status:  NOT IMPLEMENTED
    # File:    src/diarization/engine.py  (same file as Stage 5)
    # ------------------------------------------------------------------
    log.info("Stage 6 — Retroactive Re-scoring")
    raise NotImplementedError(
        "Stage 6 (Retroactive Re-scoring) is not yet implemented. "
        "See Phase 1 spec Stage 6 and src/diarization/engine.py."
    )

    # ------------------------------------------------------------------
    # Stage 7 — Output Assembly
    # Status:  NOT IMPLEMENTED
    # File:    src/utils/schema.py
    # ------------------------------------------------------------------
    log.info("Stage 7 — Output Assembly")
    raise NotImplementedError(
        "Stage 7 (Output Assembly) is not yet implemented. "
        "See Phase 1 spec Stage 7 and src/utils/schema.py."
    )

    # ------------------------------------------------------------------
    # Write output files
    # All writes are guarded by dry_run.
    # ------------------------------------------------------------------
    if dry_run:
        log.info("Dry run — skipping file writes.")
        return

    # TODO: Replace with real schema.write_outputs() call once Stage 7
    # is implemented:
    #
    # from src.utils.schema import write_outputs
    # write_outputs(
    #     incident_id=incident_id,
    #     output_dir=output_dir,
    #     result=...,
    #     write_vault=not no_vault,
    #     write_timeline=not no_timeline,
    # )