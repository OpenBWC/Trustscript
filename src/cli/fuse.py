# ==============================================================================
# src/cli/fuse.py
# ==============================================================================
"""
TrustScript CLI — Phase 4: Multi-Source Confidence Fusion  [STUB]
"""

import click
from .main import configure_logging


@click.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True))
@click.option("--output", "-o", default=None,
              type=click.Path(file_okay=False, writable=True))
@click.option("--verbose", "-v", is_flag=True, default=False)
def fuse(input_path, output, verbose) -> None:
    """
    Phase 4 — Multi-Source Confidence Fusion.  [NOT IMPLEMENTED]

    Fuses confidence signals from Phase 1 (reid_score, concurrency_confidence),
    Phase 2 (HNR, acoustic_dominance, proximity_ratio, identity_blur), and
    Phase 3 (Whisper token probabilities, Kaldi posteriors) into a composite
    confidence score per word and segment.

    Requires Phase 1, Phase 2, and Phase 3 output.
    """
    configure_logging(verbose)
    raise NotImplementedError(
        "Phase 4 (Confidence Fusion) is not yet implemented. "
        "See Phase 4 spec and src/fusion/."
    )

