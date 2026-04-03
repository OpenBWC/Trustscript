# ==============================================================================
# src/cli/transcribe.py
# ==============================================================================
"""
TrustScript CLI — Phase 3: Automatic Speech Recognition  [STUB]
"""

import click
from .main import configure_logging


@click.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Path to a Phase 1/2 output directory.")
@click.option("--output", "-o", default=None,
              type=click.Path(file_okay=False, writable=True))
@click.option("--whisper-model",
              type=click.Choice(["large-v3", "medium", "small"]),
              default="large-v3", show_default=True,
              help="Whisper model size. large-v3 recommended for forensic use.")
@click.option("--transcribe-undiarizable",
              type=click.Choice(["attempt", "skip"]),
              default="attempt", show_default=True,
              help="ASR behavior on UNDIARIZABLE regions from Phase 2.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def transcribe(
    input_path, output, whisper_model, transcribe_undiarizable, verbose,
) -> None:
    """
    Phase 3 — Automatic Speech Recognition (Whisper + Kaldi).  [NOT IMPLEMENTED]

    Runs dual-engine ASR on speaker segments from the Phase 1 timeline.
    Produces word-level transcripts with per-word confidence scores from
    Whisper token probabilities and Kaldi phoneme posteriors.

    Requires Phase 1 output. Phase 2 output is used when available.
    """
    configure_logging(verbose)
    raise NotImplementedError(
        "Phase 3 (ASR) is not yet implemented. "
        "See Phase 3 spec and src/asr/."
    )

