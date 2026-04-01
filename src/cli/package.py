# ==============================================================================
# src/cli/package.py
# ==============================================================================
"""
TrustScript CLI — Phase 5: Output Package Assembly  [STUB]
"""

import click
from .main import configure_logging


@click.command(name="package")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True))
@click.option("--output", "-o", default=None,
              type=click.Path(file_okay=False, writable=True))
@click.option("--verbose", "-v", is_flag=True, default=False)
def package_output(input_path, output, verbose) -> None:
    """
    Phase 5 — TrustScript Output Package Assembly.  [NOT IMPLEMENTED]

    Assembles all phase outputs into the final TrustScript output:
    aligned transcript with speaker labels, word-level timestamps,
    composite confidence scores, and interpretability artifacts
    (Kaldi lattice excerpts, confidence heatmaps, flagged segments).

    Requires all prior phases to have completed.
    """
    configure_logging(verbose)
    raise NotImplementedError(
        "Phase 5 (Output Package) is not yet implemented. "
        "See Phase 5 spec and src/utils/schema.py."
    )

