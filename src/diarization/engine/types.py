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
from typing import NamedTuple

import numpy as np

from ..segment import TimelineSegment
from ..vault import SpeakerVault


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Overlap between adjacent chunks. NOT configurable — correctness requirement.
#: See engine.py module docstring: Seam management strategy.
CHUNK_OVERLAP: float = 10.0  # seconds

#: Half of CHUNK_OVERLAP. Trimmed symmetrically from each non-edge boundary.
SEAM_HALF: float = CHUNK_OVERLAP / 2.0  # 5.0 seconds

#: Default chunk duration for production BWC footage.
DEFAULT_CHUNK_SIZE: float = 300.0  # seconds

#: Minimum segment duration to process. Below this, pyannote has produced a
#: boundary sliver — no reliable embedding can be extracted.
MIN_SEGMENT_DURATION: float = 0.1  # seconds

#: Minimum waveform / embedding L2 norm below which the signal is treated
#: as silence or corruption. Prevents NaN cosine distances downstream.
ZERO_VECTOR_THRESHOLD: float = 1e-8


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

    Produced by passes._collect_raw_segments().
    Consumed by candidates.py and passes._run_vault_matching().

    candidate_embeddings is always [] at construction time. The live
    Gate 3 candidate pool is the candidates_by_label dict managed by
    candidates.py — _RawSegment.candidate_embeddings is never read
    after Pass 1 builds that dict.

    Attributes
    ----------
    segment : TimelineSegment
        Partially initialized. speaker="UNASSIGNED" until Pass 2.
    embedding : np.ndarray
        512-d ECAPA-TDNN embedding.
    rms : float | None
        RMS energy. None if computation failed or waveform was silent.
    candidate_embeddings : list[np.ndarray]
        Always [] — see note above.
    overlap_local_labels : list[str]
        Local labels of co-active speakers (e.g. ["SPEAKER_01"]).
        Converted to global IDs after vault matching in Pass 2.
    """
    segment: TimelineSegment
    embedding: np.ndarray
    rms: float | None
    candidate_embeddings: list[np.ndarray]
    overlap_local_labels: list[str]