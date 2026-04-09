"""
src/diarization/engine/passes.py
===================================
The two-pass per-chunk processing loop.

Responsibility
--------------
    collect_raw_segments()  — Pass 1
        Iterates pyannote's diarization annotation. For each segment
        in the chunk's core window: applies seam filtering, detects
        overlap, extracts the embedding and RMS, and computes
        concurrency_confidence. Builds _RawSegment carriers.
        Does NOT modify vault state.

    run_vault_matching()    — Pass 2
        Calls vault.match_or_create() for each _RawSegment in order.
        After all assignments are made, resolves overlap_speakers from
        chunk-local labels to global IDs using the label→global map
        built during this pass.

Why two passes?
---------------
Pass 1 must complete before Pass 2 can begin because:
    1. candidates.build_candidates_by_label() needs all non-overlap
       segments across the whole chunk to build stable Gate 3 pools.
    2. candidates.resolve_local_conflicts() needs all per-label pools
       to score mean embeddings and detect vault collisions.

If vault.match_or_create() ran inside the Pass 1 loop, both of these
would evaluate partial evidence and produce unreliable Gate 3 results.

The two-pass design is what separates this from a naive implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..segment import (
    CONCURRENCY_CONFIRMED_THRESHOLD,
    FLAG_CONCURRENT_SPEECH,
    FLAG_GHOST_SPEAKER,
    TimelineSegment,
)
from ..vault import SpeakerVault
from .audio import compute_rms, extract_segment_embedding
from .overlap import detect_overlap
from .posteriors import concurrency_confidence_for_segment
from .types import MIN_SEGMENT_DURATION, _RawSegment

if TYPE_CHECKING:
    from pyannote.audio import Inference
    from pyannote.core import Annotation, SlidingWindowFeature, Timeline

logger = logging.getLogger(__name__)


def collect_raw_segments(
    diarization: "Annotation",
    chunk_waveform,
    sample_rate: int,
    chunk_start: float,
    chunk_idx: int,
    core_start: float,
    core_end: float,
    overlap_timeline: "Timeline | None",
    posteriors: "SlidingWindowFeature | None",
    overlap_class_indices: list[int],
    embed_inference: "Inference",
    segment_id_offset: int,
) -> list[_RawSegment]:
    """
    Pass 1: extract all segment data for this chunk's core window.

    For each turn in the diarization annotation:
        1. Translate to absolute file time (chunk_start + local_time).
        2. Apply seam guard: accept only if segment midpoint falls in
           [core_start, core_end]. Midpoint ownership prevents double-
           counting while correctly handling long straddle segments.
        3. Detect overlap via overlap.detect_overlap().
        4. Extract 512-d embedding via audio.extract_segment_embedding().
           Skip segment if extraction fails (None return).
        5. Compute RMS via audio.compute_rms().
        6. Compute concurrency_confidence from powerset posteriors
           (overlap segments only, when posteriors are available).
        7. Set overlap flags (CONCURRENT_SPEECH, GHOST_SPEAKER).

    Does NOT modify vault state.

    Parameters
    ----------
    diarization : Annotation
        pyannote diarization result for the chunk.
    chunk_waveform : torch.Tensor
        Full chunk waveform, shape (1, num_chunk_samples).
    sample_rate : int
        Audio sample rate.
    chunk_start : float
        Absolute start time of this chunk in the original file.
    chunk_idx : int
        Zero-based chunk index (for logging and TimelineSegment.chunk_index).
    core_start, core_end : float
        Accepted midpoint range in absolute file time.
    overlap_timeline : Timeline | None
        Precomputed overlap regions from diarization.get_overlap().
    posteriors : SlidingWindowFeature | None
        Powerset posteriors for concurrency_confidence. None disables it.
    overlap_class_indices : list[int]
        Powerset class indices for concurrent speech. [] disables it.
    embed_inference : Inference
        pyannote Inference object for embedding extraction.
    segment_id_offset : int
        Base segment_id for this chunk (cumulative across all chunks).

    Returns
    -------
    list[_RawSegment]
        All accepted segments from Pass 1, in annotation order.
        candidate_embeddings is [] for all — populated by candidates.py.
    """
    raw: list[_RawSegment] = []

    for turn, _, local_speaker in diarization.itertracks(yield_label=True):
        # Translate to absolute file time.
        abs_start = chunk_start + float(turn.start)
        abs_end = chunk_start + float(turn.end)
        duration = abs_end - abs_start

        # Skip pyannote micro-segments.
        if duration < MIN_SEGMENT_DURATION:
            continue

        # Seam guard: midpoint must fall within the core window.
        midpoint = (abs_start + abs_end) / 2.0
        if midpoint < core_start or midpoint > core_end:
            logger.debug(
                "Chunk %d: seam discard — %s [%.2f, %.2f] midpoint=%.2f "
                "outside core [%.2f, %.2f]",
                chunk_idx, local_speaker, abs_start, abs_end,
                midpoint, core_start, core_end,
            )
            continue

        # Overlap detection.
        is_overlap, overlap_local_labels = detect_overlap(
            turn=turn,
            local_speaker=local_speaker,
            diarization=diarization,
            overlap_timeline=overlap_timeline,
        )

        # Embedding extraction — skip segment if it fails.
        embedding = extract_segment_embedding(
            embed_inference=embed_inference,
            chunk_waveform=chunk_waveform,
            sample_rate=sample_rate,
            seg_start_local=float(turn.start),
            seg_end_local=float(turn.end),
        )
        if embedding is None:
            logger.debug(
                "Chunk %d: embedding failed for %s [%.2f, %.2f] — skipping.",
                chunk_idx, local_speaker, abs_start, abs_end,
            )
            continue

        rms = compute_rms(
            chunk_waveform, sample_rate,
            float(turn.start), float(turn.end),
        )

        # concurrency_confidence from powerset posteriors.
        concurrency_confidence: float | None = None
        if is_overlap and posteriors is not None and overlap_class_indices:
            concurrency_confidence = concurrency_confidence_for_segment(
                posteriors=posteriors,
                overlap_class_indices=overlap_class_indices,
                seg_start_local=float(turn.start),
                seg_end_local=float(turn.end),
            )

        # Overlap flags.
        flags: list[str] = []
        if is_overlap:
            flags.append(FLAG_CONCURRENT_SPEECH)
            if (
                concurrency_confidence is None
                or concurrency_confidence < CONCURRENCY_CONFIRMED_THRESHOLD
            ):
                flags.append(FLAG_GHOST_SPEAKER)

        seg = TimelineSegment(
            segment_id=segment_id_offset + len(raw),
            start_seconds=abs_start,
            end_seconds=abs_end,
            duration_seconds=duration,
            speaker="UNASSIGNED",
            local_speaker_label=local_speaker,
            reid_score=None,
            overlap=is_overlap,
            overlap_speakers=[],          # Resolved after Pass 2.
            concurrency_confidence=concurrency_confidence,
            flags=flags,
            chunk_index=chunk_idx,
        )

        raw.append(_RawSegment(
            segment=seg,
            embedding=embedding,
            rms=rms,
            overlap_local_labels=overlap_local_labels,
        ))

    return raw


def run_vault_matching(
    raw_segments: list[_RawSegment],
    candidates_by_label: dict,
    vault: SpeakerVault,
) -> list[TimelineSegment]:
    """
    Pass 2: call vault.match_or_create() for each segment in order.

    Builds a label→global mapping as assignments are made. After all
    segments are matched, resolves overlap_speakers from chunk-local
    labels (stored in _RawSegment.overlap_local_labels) to global IDs
    using that mapping.

    Segments whose speaker could not be resolved (UNKNOWN, UNASSIGNED)
    are not added to label_to_global, so their local labels appear
    verbatim in overlap_speakers as a fallback. This preserves
    auditability without crashing on unresolved labels.

    Parameters
    ----------
    raw_segments : list[_RawSegment]
        Output of collect_raw_segments().
    candidates_by_label : dict[str, list[np.ndarray]]
        Gate 3 candidates after conflict resolution from candidates.py.
    vault : SpeakerVault
        Mutated in place by match_or_create().

    Returns
    -------
    list[TimelineSegment]
        Matched segments with global speaker IDs and confidence flags.
    """
    matched: list[TimelineSegment] = []
    label_to_global: dict[str, str] = {}

    for raw in raw_segments:
        seg = raw.segment
        candidates = candidates_by_label.get(seg.local_speaker_label, [])

        vault.match_or_create(
            segment=seg,
            embedding=raw.embedding,
            rms=raw.rms,
            candidate_embeddings=candidates,
        )

        # Record confirmed global IDs for overlap_speakers resolution.
        if seg.speaker not in ("UNKNOWN", "UNASSIGNED", "OVERLAP"):
            label_to_global[seg.local_speaker_label] = seg.speaker

        matched.append(seg)

    # Resolve overlap_speakers: local labels → global IDs.
    # Runs after all segments so label_to_global is fully populated.
    for raw in raw_segments:
        if raw.overlap_local_labels:
            raw.segment.overlap_speakers = [
                label_to_global.get(lbl, lbl)
                for lbl in raw.overlap_local_labels
            ]

    return matched