# ==============================================================================
# src/cli/overlap.py
# ==============================================================================
"""
TrustScript CLI — Phase 2: Overlap Characterization and Triage  [STUB]

Owns the `trustscript overlap` command. All flags are defined and
documented. The command raises NotImplementedError until Phase 2 is
implemented in src/overlap/.
"""

import click
from .main import configure_logging, console


@click.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Path to a Phase 1 output directory or _phase1.json file.")
@click.option("--output", "-o", default=None,
              type=click.Path(file_okay=False, writable=True),
              help="Directory to write Phase 2 output files.")
@click.option("--undiarizable-duration", default=30, show_default=True,
              type=click.IntRange(min=5),
              help="Duration (seconds) above which an overlap is UNDIARIZABLE.")
@click.option("--undiarizable-density", default=0.70, show_default=True,
              type=click.FloatRange(min=0.1, max=1.0),
              help="Concurrent speech density above which an overlap is UNDIARIZABLE.")
@click.option("--undiarizable-hnr", default=5.0, show_default=True,
              type=float,
              help="HNR (dB) below which an overlap is a candidate for UNDIARIZABLE.")
@click.option("--short-overlap-ceiling", default=3.0, show_default=True,
              type=float,
              help="Duration (seconds) below which overlaps are OVERLAP_SHORT.")
@click.option("--ghost-speaker-threshold", default=0.65, show_default=True,
              type=click.FloatRange(min=0.0, max=1.0),
              help="concurrency_confidence below this value is GHOST_SPEAKER.")
@click.option("--transcribe-undiarizable",
              type=click.Choice(["attempt", "skip"]), default="attempt",
              show_default=True,
              help=(
                  "Phase 3 ASR behavior on UNDIARIZABLE regions. "
                  "'attempt': transcribe with no speaker attribution. "
                  "'skip': omit from transcript entirely."
              ))
@click.option("--verbose", "-v", is_flag=True, default=False)
def overlap(
    input_path, output, undiarizable_duration, undiarizable_density,
    undiarizable_hnr, short_overlap_ceiling, ghost_speaker_threshold,
    transcribe_undiarizable, verbose,
) -> None:
    """
    Phase 2 — Overlap Characterization and Triage.  [NOT IMPLEMENTED]

    Characterizes overlap regions identified by Phase 1 using acoustic
    metrics (HNR, F0, spectral contrast, acoustic dominance) and assigns
    each region a triage category: OVERLAP_SHORT, OVERLAP_LONG,
    UNDIARIZABLE, or GHOST_SPEAKER.

    Requires Phase 1 output. Run `trustscript diarize` first.
    """
    configure_logging(verbose)
    raise NotImplementedError(
        "Phase 2 (Overlap Characterization) is not yet implemented. "
        "See Phase 2 spec and src/overlap/."
    )

