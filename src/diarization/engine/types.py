"""
src/diarization/engine/types.py
=================================
Engine-wide constants, result types, and internal data structures.

This module has zero internal dependencies — it imports only from
the standard library, numpy, and the parent diarization package.
Every other engine module imports from here; nothing here imports
from other engine modules.

Constants
---------
    CHUNK_OVERLAP           10-second boundary overlap (not configurable).
    SEAM_HALF               5-second trim applied to each non-edge boundary.
    DEFAULT_CHUNK_SIZE      300s production default.
    MIN_SEGMENT_DURATION    Minimum pyannote segment to process (0.1s).
    ZERO_VECTOR_THRESHOLD   Norm floor for silence/corruption detection.

Public types
------------
    EngineResult    Returned by run_windowed_diarization(). Consumed by
                    stage5.py for timeline assembly and refined SNR.

Internal types
--------------
    _RawSegment     NamedTuple. Carries per-segment data between Pass 1
                    (extraction) and Pass 2 (vault matching). Immutable
                    after construction — the candidate_embeddings field
                    is always [] here; the live candidates dict is managed
                    by candidates.py separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, NamedTuple

import numpy as np

from ..segment import TimelineSegment
from ..vault import SpeakerVault


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Overlap between adjacent chunks. NOT configurable — correctness requirement.
#: See engine.py module docstring: Seam management strategy.
CHUNK_OVERLAP: Final[float] = 10.0  # seconds

#: Half of CHUNK_OVERLAP. Trimmed symmetrically from each non-edge boundary.
SEAM_HALF: Final[float] = CHUNK_OVERLAP / 2.0  # 5.0 seconds

#: Default chunk duration for production BWC footage.
DEFAULT_CHUNK_SIZE: Final[float] = 300.0  # seconds

#: Minimum segment duration to process. Below this, pyannote has produced a
#: boundary sliver — no reliable embedding can be extracted.
MIN_SEGMENT_DURATION: Final[float] = 0.1  # seconds

#: Minimum waveform / embedding L2 norm below which the signal is treated
#: as silence or corruption. Prevents NaN cosine distances downstream.
ZERO_VECTOR_THRESHOLD: Final[float] = 1e-8

#: Canonical dtype for all ECAPA-TDNN embeddings throughout the engine.
#: float32 is enforced at extraction time in audio.py and must be preserved
#: through vault storage, cosine matching, and centroid updates.
#: Accidentally storing float64 doubles RAM per embedding; float16 causes
#: non-deterministic behaviour in Phase 4 Fusion's cosine arithmetic.
EMBEDDING_DTYPE: Final[type] = np.float32


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """
    Complete output of run_windowed_diarization(). Passed to stage5.py.

    Attributes
    ----------
    timeline : list[TimelineSegment]
        All accepted segments in chronological order.
        Speaker IDs are global (TRUST_SPK_NN).
    vault : SpeakerVault
        Fully populated vault after all chunks are processed.
    chunk_count : int
        Number of chunks processed.
    total_speech_seconds : float
        Sum of non-overlap segment durations. Used for reporting.
    speech_intervals : list[tuple[float, float]]
        (start, end) pairs from pyannote VAD output (non-overlap segments).
        Used by stage5.py to compute refined SNR after the full run.
    """
    timeline: list[TimelineSegment]
    vault: SpeakerVault
    chunk_count: int
    total_speech_seconds: float
    speech_intervals: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal per-segment carrier (Pass 1 → Pass 2)
# ---------------------------------------------------------------------------

class _RawSegment(NamedTuple):
    """
    Immutable carrier for one segment's Pass 1 data.

    Produced by passes.collect_raw_segments().
    Consumed by candidates.py and passes.run_vault_matching().

    candidate_embeddings is intentionally absent. The Gate 3 pool
    is a chunk-level structure (local_label → list[embeddings]) managed
    by candidates.build_candidates_by_label() and passed separately to
    run_vault_matching(). Carrying an always-empty list on every segment
    object would add overhead across thousands of segments in a long file
    with no consumer to justify the cost.

    All embeddings stored here are dtype=EMBEDDING_DTYPE (float32).
    See EMBEDDING_DTYPE for the rationale.

    Attributes
    ----------
    segment : TimelineSegment
        Partially initialized. speaker="UNASSIGNED" until Pass 2.
    embedding : np.ndarray
        512-d ECAPA-TDNN embedding, dtype float32.
    rms : float | None
        RMS energy. None if computation failed or waveform was silent.
        0.0 is a valid value (digital silence) — callers use
        `if rms is not None`, never `if rms`.
    overlap_local_labels : list[str]
        Local labels of co-active speakers (e.g. ["SPEAKER_01"]).
        Converted to global IDs after vault matching in Pass 2.
    """
    segment: TimelineSegment
    embedding: np.ndarray
    rms: float | None
    overlap_local_labels: list[str]