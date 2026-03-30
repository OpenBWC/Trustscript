"""
src/utils/audio/hashing.py
===========================
TrustScript Phase 1 — Chain-of-Custody Hashing

Responsibility
--------------
Compute SHA256 hashes of input files before any processing occurs.

This module has exactly one job: given a file path, return a SHA256
digest. It does not read audio properties, spawn ffmpeg processes,
or perform any validation beyond what is needed to open the file.

Why SHA256 and why before processing?
--------------------------------------
TrustScript's chain-of-custody requirement demands that the integrity
of the original source material be provable at any future point, even
after the original file is no longer present (e.g. deleted after
processing, or stored separately from pipeline outputs).

The hash is computed on the original file exactly as submitted —
video container and all — before audio extraction, normalization,
or any other modification. This means:

    - For .mp4 inputs: the hash covers the full video file, not just
      the extracted audio stream. This records what was actually
      submitted to the pipeline.
    - For .wav inputs: the hash covers the audio file as provided.

The hash is written to the `processing_notes.stage_1.original_sha256`
field in every TrustScript output JSON.

Contributors
------------
    Do not add audio processing logic to this module. If you need to
    hash something other than a file, add a separate function here
    rather than mixing concerns with probe.py or stage1.py.
"""

import hashlib
import logging
from pathlib import Path

from .exceptions import AudioIngestionError

logger = logging.getLogger(__name__)

# Read files in 64KB chunks which large enough for throughput efficiency,
# small enough to process arbitrarily large video files without loading
# them entirely into memory.
_HASH_CHUNK_SIZE: int = 64 * 1024


def compute_sha256(path: Path) -> str:
    """
    Compute the SHA256 hash of a file for chain-of-custody provenance.

    This function must be called before any other processing on the file.
    It is the first substantive operation in Stage 1 after path validation,
    and it records the unmodified state of the original source material.

    The file is read in chunks to handle large video containers without
    loading the entire file into memory.

    Parameters
    ----------
    path : Path
        Path to the file to hash. Must exist and be readable.
        Relative paths should be resolved before calling this function.

    Returns
    -------
    str
        Lowercase hexadecimal SHA256 digest. Always 64 characters.

    Raises
    ------
    AudioIngestionError
        If the file cannot be opened or read due to permissions or
        an I/O error.

    Examples
    --------
    >>> digest = compute_sha256(Path("footage/incident_001.mp4"))
    >>> len(digest)
    64
    >>> digest.islower()
    True
    """
    hasher = hashlib.sha256()

    try:
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK_SIZE):
                hasher.update(chunk)
    except OSError as e:
        raise AudioIngestionError(
            f"Could not read file for SHA256 hashing: {path}\n"
            f"OS error: {e}"
        ) from e

    digest = hasher.hexdigest()
    logger.debug("SHA256 for %s: %s", path.name, digest)
    return digest