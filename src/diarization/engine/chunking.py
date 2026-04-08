"""
src/diarization/engine/chunking.py
=====================================
Chunk window generation and core window computation.

Responsibility
--------------
Two pure functions with no side effects:

    generate_chunk_windows()
        Divides audio_duration into overlapping (start, end) pairs.
        Adjacent chunks share CHUNK_OVERLAP = 10.0 seconds at every
        boundary. The last chunk extends exactly to audio_duration.

    compute_core_window()
        Returns the accepted region for one chunk in absolute file time.
        Trims SEAM_HALF = 5.0 seconds from each non-edge boundary.
        Segments whose midpoint falls outside the core window are
        discarded by passes._collect_raw_segments() and covered by
        the adjacent chunk.

No I/O, no model calls, no vault access.
"""

from __future__ import annotations

from .types import CHUNK_OVERLAP, SEAM_HALF


def generate_chunk_windows(
    audio_duration: float,
    chunk_size: float,
) -> list[tuple[float, float]]:
    """
    Generate overlapping (start, end) pairs covering the full audio.

    Adjacent chunks overlap by CHUNK_OVERLAP = 10.0 seconds. The seam
    management in compute_core_window() resolves ownership of segments
    near each boundary — the overlap itself is what gives the vault
    enough signal to confirm identity before committing cross-seam
    assignments.

    The last chunk extends to audio_duration exactly, which may make
    it shorter than chunk_size.

    Example — 12-minute audio, 5-minute chunks, 10s overlap:
        [(0.0, 300.0), (290.0, 590.0), (580.0, 720.0)]

    Parameters
    ----------
    audio_duration : float
        Total file duration in seconds.
    chunk_size : float
        Requested chunk duration in seconds.

    Returns
    -------
    list[tuple[float, float]]
        Ordered list of (chunk_start, chunk_end) in seconds.
    """
    if chunk_size <= CHUNK_OVERLAP:
        raise ValueError(
            f"chunk_size ({chunk_size}s) must be greater than "
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}s). "
            f"At or below this value, chunk start never advances and "
            f"the window loop will not terminate. "
            f"Minimum valid chunk_size is {CHUNK_OVERLAP + 1.0}s."
        )

    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < audio_duration:
        end = min(start + chunk_size, audio_duration)
        windows.append((start, end))
        if end >= audio_duration:
            break
        start = end - CHUNK_OVERLAP
    return windows


def compute_core_window(
    chunk_start: float,
    chunk_end: float,
    is_first: bool,
    is_last: bool,
) -> tuple[float, float]:
    """
    Compute the accepted core window for one chunk in absolute file time.

    Trims SEAM_HALF seconds from each non-edge boundary. Segments whose
    midpoint falls outside [core_start, core_end] are discarded —
    they are covered by the adjacent chunk's core window.

    Midpoint ownership rule:
        A segment's midpoint determines which chunk claims it.
        Long segments straddling the seam are assigned to the chunk
        whose core window contains their center.

    Edge rules:
        First chunk  — no left trim (nothing to the left).
        Last chunk   — no right trim (nothing to the right).
        Single chunk — no trim on either side (both flags True).

    Parameters
    ----------
    chunk_start, chunk_end : float
        Absolute file-time boundaries of this chunk (seconds).
    is_first, is_last : bool
        Whether this is the first / last chunk in the sequence.

    Returns
    -------
    tuple[float, float]
        (core_start, core_end) in absolute file time.
    """
    core_start = chunk_start if is_first else chunk_start + SEAM_HALF
    core_end = chunk_end if is_last else chunk_end - SEAM_HALF
    return core_start, core_end