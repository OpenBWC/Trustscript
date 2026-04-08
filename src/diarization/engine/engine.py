"""
src/diarization/engine.py
==========================
TrustScript Phase 1 — Stage 5: Windowed Diarization Engine

Responsibility
--------------
Run pyannote diarization over the full normalized audio file in
configurable overlapping chunks. For each chunk: extract speaker
embeddings, detect concurrent speech with confidence scores, and
match speakers against the global Speaker Anchor Vault.

This is the most computationally dangerous file in Phase 1.
It interfaces with C-extension audio I/O (torchaudio / libsoundfile),
pyannote's inference pipeline, and PyTorch memory management. The
failure modes are silent RAM exhaustion, NaN propagation from zero-
vector embeddings, and double-counting at chunk seams.

Hardware guardrails
-------------------
ALL inference is wrapped in `torch.no_grad()`. Without this, PyTorch
builds a backpropagation graph for every forward pass. On an 8GB
machine processing 5-minute chunks, this exhausts RAM within the first
two chunks and produces an OOM crash with no useful error message.

torch.no_grad() is applied at the outermost loop level, not per-call.
This is intentional — the guard must cover the full chunk lifecycle
including torchaudio loads and embedding extraction, not just the
pyannote call.

Internal API access: pipeline._segmentation
---------------------------------------------
concurrency_confidence is extracted by calling pipeline._segmentation()
directly on the chunk waveform after pipeline() has already returned
the diarization annotation. This is a second forward pass through the
segmentation model only (not the full diarization pipeline).

    posteriors = pipeline._segmentation(audio_input)
    # → SlidingWindowFeature of shape (num_frames, num_powerset_classes)

This API is internal to pyannote.audio and not part of its public
interface. It is documented here and in _get_overlap_class_indices()
because it is the only way to access the raw powerset softmax
probabilities that power concurrency_confidence. Pinned to
pyannote.audio >= 3.0 / Community-1 weights.

If the internal API breaks between pyannote versions, the engine
degrades gracefully: concurrency_confidence is set to None for all
segments, all overlap detections still work via diarization.get_overlap(),
and the rest of the pipeline is unaffected.

Seam management strategy
------------------------
Adjacent chunks overlap by CHUNK_OVERLAP = 10.0 seconds. A segment
near a boundary appears in both chunks. The core-window strategy
resolves ownership:

    core_start = chunk_start + SEAM_HALF  (except first chunk)
    core_end   = chunk_end   - SEAM_HALF  (except last chunk)
    SEAM_HALF  = 5.0 seconds

A segment is accepted into the timeline only if its MIDPOINT falls
within [core_start, core_end]. Midpoint ownership (rather than start
or end) correctly handles long segments that straddle the seam —
the chunk covering the majority of the segment claims it.

Two-pass design
---------------
Each chunk is processed in two passes to enable correct Gate 3 and
local conflict handling:

Pass 1 (_collect_raw_segments):
    Iterate pyannote annotation. Extract embeddings, RMS, overlap
    flags, and concurrency_confidence for every segment in the core
    window. No vault state is modified.

Pass 2 (_run_vault_matching):
    After _build_candidates_by_label and _resolve_local_conflicts have
    prepared the Gate 3 evidence, call vault.match_or_create() for
    each segment in order. Vault state is modified here and only here.

This separation is required because _resolve_local_conflicts needs the
full chunk's embeddings to detect conflicts, and _build_candidates_by_label
needs all non-overlap segments per label before any vault writes occur.

Local conflict check
---------------------
If two chunk-local labels (e.g. SPEAKER_00 and SPEAKER_01) both score
above MATCH_THRESHOLD against the same vault entry, pyannote has split
one person's voice into two local IDs. Resolution: pool their
candidate_embeddings for Gate 3. The vault will naturally assign both
to the same global ID via cosine matching. Pooling prevents false
INSUFFICIENT_SEGMENTS rejections where each label alone has < N=3
clean segments but together they have enough evidence.

Contributors
------------
    Stage 4 model loading belongs in models.py.
    Vault gate logic belongs in gates.py.
    Vault state management belongs in vault/.
    Do not add Stage 6 re-scoring logic here.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torchaudio

from ..models import Stage4Result
from ..segment import (
    CONCURRENCY_CONFIRMED_THRESHOLD,
    FLAG_CONCURRENT_SPEECH,
    FLAG_GHOST_SPEAKER,
    TimelineSegment,
)
from ..vault import SpeakerVault
from ..vault.matching import find_best_match
from ..vault.types import MATCH_THRESHOLD

if TYPE_CHECKING:
    from pyannote.audio import Inference
    from pyannote.core import Annotation, Segment as PySegment, Timeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Overlap between adjacent chunks. NOT configurable — correctness requirement.
#: A speaker mid-sentence at a boundary appears in both chunks; the overlap
#: region gives the vault enough signal to confirm identity before committing
#: cross-seam assignments.
CHUNK_OVERLAP: float = 10.0  # seconds

#: Half of CHUNK_OVERLAP. Trimmed symmetrically from each non-edge chunk boundary.
SEAM_HALF: float = CHUNK_OVERLAP / 2.0  # 5.0 seconds

#: Default chunk duration for production BWC footage.
DEFAULT_CHUNK_SIZE: float = 300.0  # seconds

#: Minimum segment duration to process. Below this, pyannote has produced a
#: boundary sliver — no reliable embedding can be extracted, and attempting
#: one wastes compute and risks poisoning vault candidates.
MIN_SEGMENT_DURATION: float = 0.1  # seconds

#: Minimum waveform L2 norm below which a segment is treated as silence.
#: Prevents zero-vector embeddings that produce NaN cosine distances.
ZERO_VECTOR_THRESHOLD: float = 1e-8


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """
    Output of run_windowed_diarization(). Passed to stage5.py for assembly.

    Attributes
    ----------
    timeline : list[TimelineSegment]
        All segments accepted into the timeline, in chronological order.
        Speaker IDs are global (TRUST_SPK_NN).
    vault : SpeakerVault
        The fully populated vault after processing all chunks.
    chunk_count : int
        Number of chunks processed.
    total_speech_seconds : float
        Sum of all non-overlap segment durations. Used for reporting.
    speech_intervals : list[tuple[float, float]]
        (start, end) pairs from pyannote VAD output (non-overlap segments).
        Used by stage5.py to compute refined SNR after the full run.
    """
    timeline: list[TimelineSegment]
    vault: SpeakerVault
    chunk_count: int
    total_speech_seconds: float
    speech_intervals: list[tuple[float, float]] = field(default_factory=list)


class _RawSegment(NamedTuple):
    """
    Internal: per-segment data from Pass 1. Immutable until vault matching.

    Attributes
    ----------
    segment : TimelineSegment
        Partially initialized — speaker="UNASSIGNED" until Pass 2.
    embedding : np.ndarray
        512-d ECAPA-TDNN embedding for this segment.
    rms : float | None
        RMS energy. None if computation failed.
    candidate_embeddings : list[np.ndarray]
        Clean embeddings for this segment's local label (for Gate 3).
        Populated by _build_candidates_by_label after Pass 1.
        Mutated by _resolve_local_conflicts if a conflict is detected.
    overlap_local_labels : list[str]
        Local labels of co-active speakers (e.g. ["SPEAKER_01"]).
        Converted to global IDs after vault matching in Pass 2.
    """
    segment: TimelineSegment
    embedding: np.ndarray
    rms: float | None
    candidate_embeddings: list[np.ndarray]
    overlap_local_labels: list[str]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_windowed_diarization(
    audio_path: Path,
    stage4_result: Stage4Result,
    vault: SpeakerVault,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
) -> EngineResult:
    """
    Run windowed diarization over the full normalized audio file.

    Processes audio in overlapping chunks. For each chunk: runs pyannote,
    extracts embeddings and concurrency_confidence, and matches against
    the Speaker Anchor Vault. Returns the full timeline and populated vault.

    Parameters
    ----------
    audio_path : Path
        Normalized mono WAV from Stage 3. Must be 16kHz or higher.
    stage4_result : Stage4Result
        Loaded pyannote models from Stage 4.
    vault : SpeakerVault
        Live vault, possibly pre-seeded by stage5.py initialization logic.
        Mutated in place throughout the loop.
    chunk_size : float
        Chunk duration in seconds. Default: 300.0 (5 minutes).

    Returns
    -------
    EngineResult
        Complete timeline, vault, and VAD data for refined SNR.
    """
    pipeline = stage4_result.pipeline
    embed_inference = stage4_result.embed_inference

    # Read audio metadata once — reused for all chunk loads.
    audio_info = torchaudio.info(str(audio_path))
    sample_rate: int = audio_info.sample_rate
    audio_duration: float = audio_info.num_frames / sample_rate

    logger.info(
        "Engine: audio=%.2fs @ %dHz, chunk=%.0fs, overlap=%.0fs",
        audio_duration, sample_rate, chunk_size, CHUNK_OVERLAP,
    )

    # Compute overlap class indices once before the loop.
    # Accesses pipeline._segmentation.model (internal API — see module docstring).
    overlap_class_indices: list[int] = _get_overlap_class_indices(pipeline)
    if not overlap_class_indices:
        logger.warning(
            "Powerset overlap class indices unavailable. "
            "concurrency_confidence will be None for all segments."
        )

    chunk_windows = _generate_chunk_windows(audio_duration, chunk_size)
    num_chunks = len(chunk_windows)

    timeline: list[TimelineSegment] = []
    speech_intervals: list[tuple[float, float]] = []
    segment_counter: int = 0

    # ── Hardware guardrail ────────────────────────────────────────────────
    # torch.no_grad() wraps the entire loop. Without this, PyTorch builds
    # a backpropagation graph across all forward passes, exhausting RAM on
    # 8GB machines within the first two chunks. Applied at the outermost
    # level to cover torchaudio loads, pyannote inference, and embeddings.
    # ─────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_windows):
            is_first = chunk_idx == 0
            is_last = chunk_idx == num_chunks - 1

            logger.info(
                "Chunk %d/%d: %.1fs – %.1fs",
                chunk_idx + 1, num_chunks, chunk_start, chunk_end,
            )

            chunk_segments, chunk_speech = _process_chunk(
                audio_path=audio_path,
                sample_rate=sample_rate,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                chunk_idx=chunk_idx,
                is_first=is_first,
                is_last=is_last,
                pipeline=pipeline,
                embed_inference=embed_inference,
                vault=vault,
                overlap_class_indices=overlap_class_indices,
                segment_id_offset=segment_counter,
            )

            timeline.extend(chunk_segments)
            speech_intervals.extend(chunk_speech)
            segment_counter += len(chunk_segments)

    total_speech = sum(e - s for s, e in speech_intervals)
    logger.info(
        "Engine complete: %d segments, %.2fs speech, %d chunks processed.",
        len(timeline), total_speech, num_chunks,
    )

    return EngineResult(
        timeline=timeline,
        vault=vault,
        chunk_count=num_chunks,
        total_speech_seconds=total_speech,
        speech_intervals=speech_intervals,
    )


# ---------------------------------------------------------------------------
# Per-chunk orchestration
# ---------------------------------------------------------------------------

def _process_chunk(
    audio_path: Path,
    sample_rate: int,
    chunk_start: float,
    chunk_end: float,
    chunk_idx: int,
    is_first: bool,
    is_last: bool,
    pipeline,
    embed_inference: "Inference",
    vault: SpeakerVault,
    overlap_class_indices: list[int],
    segment_id_offset: int,
) -> tuple[list[TimelineSegment], list[tuple[float, float]]]:
    """
    Full processing pipeline for one chunk.

    Returns
    -------
    tuple[list[TimelineSegment], list[tuple[float, float]]]
        (accepted segments, speech intervals for refined SNR)
    """
    # ── Audio load ───────────────────────────────────────────────────────
    chunk_waveform = _load_chunk_audio(
        audio_path, chunk_start, chunk_end, sample_rate
    )
    audio_input = {"waveform": chunk_waveform, "sample_rate": sample_rate}

    # ── Diarization ──────────────────────────────────────────────────────
    try:
        diarization: "Annotation" = pipeline(audio_input)
    except Exception as exc:
        logger.error(
            "Chunk %d: pyannote diarization failed: %s — skipping chunk.",
            chunk_idx, exc,
        )
        return [], []

    # ── Powerset posteriors (internal API — see module docstring) ────────
    posteriors = _extract_posteriors(pipeline, audio_input)

    # ── Overlap regions ──────────────────────────────────────────────────
    overlap_timeline: "Timeline | None" = None
    try:
        overlap_timeline = diarization.get_overlap()
    except Exception as exc:
        logger.warning(
            "Chunk %d: diarization.get_overlap() failed: %s. "
            "Overlap detection will use annotation intersection fallback.",
            chunk_idx, exc,
        )

    # ── Core window for seam management ─────────────────────────────────
    core_start, core_end = _compute_core_window(
        chunk_start, chunk_end, is_first, is_last
    )
    logger.debug(
        "Chunk %d: core window [%.2fs, %.2fs]", chunk_idx, core_start, core_end
    )

    # ── Pass 1: collect all segment data ────────────────────────────────
    raw_segments = _collect_raw_segments(
        diarization=diarization,
        chunk_waveform=chunk_waveform,
        sample_rate=sample_rate,
        chunk_start=chunk_start,
        chunk_idx=chunk_idx,
        core_start=core_start,
        core_end=core_end,
        overlap_timeline=overlap_timeline,
        posteriors=posteriors,
        overlap_class_indices=overlap_class_indices,
        embed_inference=embed_inference,
        segment_id_offset=segment_id_offset,
    )

    if not raw_segments:
        logger.debug("Chunk %d: no segments in core window.", chunk_idx)
        return [], []

    # ── Gate 3 candidate preparation ─────────────────────────────────────
    candidates_by_label = _build_candidates_by_label(raw_segments)

    # ── Local conflict check (Trap 2) ─────────────────────────────────────
    candidates_by_label = _resolve_local_conflicts(
        raw_segments, candidates_by_label, vault
    )

    # ── Pass 2: vault matching ────────────────────────────────────────────
    matched_segments = _run_vault_matching(
        raw_segments=raw_segments,
        candidates_by_label=candidates_by_label,
        vault=vault,
    )

    # ── Speech intervals for refined SNR ─────────────────────────────────
    speech_intervals = [
        (seg.start_seconds, seg.end_seconds)
        for seg in matched_segments
        if not seg.overlap
    ]

    return matched_segments, speech_intervals


# ---------------------------------------------------------------------------
# Pass 1: segment extraction
# ---------------------------------------------------------------------------

def _collect_raw_segments(
    diarization: "Annotation",
    chunk_waveform: torch.Tensor,
    sample_rate: int,
    chunk_start: float,
    chunk_idx: int,
    core_start: float,
    core_end: float,
    overlap_timeline: "Timeline | None",
    posteriors,
    overlap_class_indices: list[int],
    embed_inference: "Inference",
    segment_id_offset: int,
) -> list[_RawSegment]:
    """
    Pass 1: iterate pyannote annotation and extract all segment data.

    Applies seam filtering, overlap detection, embedding extraction,
    RMS computation, and concurrency_confidence calculation. Does NOT
    modify vault state. Returns a list of _RawSegment with
    candidate_embeddings as empty lists — populated by
    _build_candidates_by_label after this function returns.
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

        # ── Seam guard ───────────────────────────────────────────────────
        # Accept only segments whose midpoint falls in [core_start, core_end].
        # Midpoint ownership prevents double-counting while correctly
        # handling long segments that straddle the boundary.
        midpoint = (abs_start + abs_end) / 2.0
        if midpoint < core_start or midpoint > core_end:
            logger.debug(
                "Chunk %d: seam discard — %s [%.2fs, %.2fs] midpoint=%.2fs "
                "outside core [%.2fs, %.2fs]",
                chunk_idx, local_speaker, abs_start, abs_end,
                midpoint, core_start, core_end,
            )
            continue

        # ── Overlap detection ─────────────────────────────────────────────
        is_overlap, overlap_local_labels = _detect_overlap(
            turn=turn,
            local_speaker=local_speaker,
            diarization=diarization,
            overlap_timeline=overlap_timeline,
        )

        # ── Embedding extraction ──────────────────────────────────────────
        embedding = _extract_segment_embedding(
            embed_inference=embed_inference,
            chunk_waveform=chunk_waveform,
            sample_rate=sample_rate,
            seg_start_local=float(turn.start),
            seg_end_local=float(turn.end),
        )
        if embedding is None:
            logger.debug(
                "Chunk %d: embedding failed for %s [%.2fs, %.2fs] — skipping.",
                chunk_idx, local_speaker, abs_start, abs_end,
            )
            continue

        # ── RMS ───────────────────────────────────────────────────────────
        rms = _compute_rms(chunk_waveform, sample_rate, float(turn.start), float(turn.end))

        # ── concurrency_confidence ────────────────────────────────────────
        concurrency_confidence: float | None = None
        if is_overlap and posteriors is not None and overlap_class_indices:
            concurrency_confidence = _concurrency_confidence_for_segment(
                posteriors=posteriors,
                overlap_class_indices=overlap_class_indices,
                seg_start_local=float(turn.start),
                seg_end_local=float(turn.end),
            )

        # ── Overlap flags ─────────────────────────────────────────────────
        flags: list[str] = []
        if is_overlap:
            flags.append(FLAG_CONCURRENT_SPEECH)
            if (
                concurrency_confidence is None
                or concurrency_confidence < CONCURRENCY_CONFIRMED_THRESHOLD
            ):
                flags.append(FLAG_GHOST_SPEAKER)

        segment_id = segment_id_offset + len(raw)

        seg = TimelineSegment(
            segment_id=segment_id,
            start_seconds=abs_start,
            end_seconds=abs_end,
            duration_seconds=duration,
            speaker="UNASSIGNED",       # Assigned in Pass 2.
            local_speaker_label=local_speaker,
            reid_score=None,
            overlap=is_overlap,
            overlap_speakers=[],        # Resolved after Pass 2.
            concurrency_confidence=concurrency_confidence,
            flags=flags,
            chunk_index=chunk_idx,
        )

        raw.append(_RawSegment(
            segment=seg,
            embedding=embedding,
            rms=rms,
            candidate_embeddings=[],    # Populated by _build_candidates_by_label.
            overlap_local_labels=overlap_local_labels,
        ))

    return raw


# ---------------------------------------------------------------------------
# Gate 3 candidate preparation and conflict resolution
# ---------------------------------------------------------------------------

def _build_candidates_by_label(
    raw_segments: list[_RawSegment],
) -> dict[str, list[np.ndarray]]:
    """
    Collect clean (non-overlap) embeddings per local speaker label.

    These are the Gate 3 candidate pools. Gate 3 requires N >= 3 clean
    segments with tight pairwise spread before creating a new vault anchor.
    Including overlap-contaminated embeddings would corrupt this check.
    """
    candidates: dict[str, list[np.ndarray]] = defaultdict(list)
    for raw in raw_segments:
        if not raw.segment.overlap:
            candidates[raw.segment.local_speaker_label].append(raw.embedding)
    return dict(candidates)


def _resolve_local_conflicts(
    raw_segments: list[_RawSegment],
    candidates_by_label: dict[str, list[np.ndarray]],
    vault: SpeakerVault,
) -> dict[str, list[np.ndarray]]:
    """
    Detect and resolve local speaker label conflicts within a chunk.

    A conflict occurs when two chunk-local labels both score above
    MATCH_THRESHOLD for the same vault entry. This is pyannote splitting
    one speaker into two local IDs — a known failure at noisy boundaries.

    Resolution: pool the conflicting labels' candidate_embeddings together.
    Both labels will naturally match the same vault ID via cosine scoring.
    Pooling ensures Gate 3 evaluates their combined evidence, preventing
    false INSUFFICIENT_SEGMENTS rejections when each label alone has < N
    clean segments but together they do.

    No centroid writes occur here — this is pre-matching preparation only.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Updated candidates_by_label with pooled embeddings for conflicts.
    """
    if not vault.anchors:
        return candidates_by_label

    # Score each label's mean embedding against the vault.
    label_to_vault_match: dict[str, tuple[str, float]] = {}
    for label, embs in candidates_by_label.items():
        if not embs:
            continue
        mean_emb = np.mean(np.stack(embs), axis=0)
        best_id, best_score, _ = find_best_match(mean_emb, vault.anchors)
        if best_id is not None and best_score >= MATCH_THRESHOLD:
            label_to_vault_match[label] = (best_id, best_score)

    # Group labels by their matched vault ID.
    vault_id_to_labels: dict[str, list[str]] = defaultdict(list)
    for label, (vault_id, _) in label_to_vault_match.items():
        vault_id_to_labels[vault_id].append(label)

    updated = dict(candidates_by_label)

    for vault_id, conflicting in vault_id_to_labels.items():
        if len(conflicting) <= 1:
            continue

        # Sort by score descending — winner has highest cosine similarity.
        conflicting_sorted = sorted(
            conflicting,
            key=lambda lbl: label_to_vault_match[lbl][1],
            reverse=True,
        )
        winner = conflicting_sorted[0]
        logger.warning(
            "Local conflict on vault %s: labels %s — winner: %s (score=%.4f). "
            "Pooling candidate_embeddings for Gate 3.",
            vault_id, conflicting, winner, label_to_vault_match[winner][1],
        )

        # Pool all conflicting labels' candidates under every conflicting label.
        # Each label now sees the full evidence set for Gate 3 evaluation.
        pooled: list[np.ndarray] = []
        for lbl in conflicting:
            pooled.extend(updated.get(lbl, []))
        for lbl in conflicting:
            updated[lbl] = pooled

    return updated


# ---------------------------------------------------------------------------
# Pass 2: vault matching
# ---------------------------------------------------------------------------

def _run_vault_matching(
    raw_segments: list[_RawSegment],
    candidates_by_label: dict[str, list[np.ndarray]],
    vault: SpeakerVault,
) -> list[TimelineSegment]:
    """
    Pass 2: call vault.match_or_create() for each segment in order.

    After all segments are matched, resolve overlap_speakers from local
    labels (stored in _RawSegment.overlap_local_labels) to global IDs
    using the label→global mapping built during this pass.

    Parameters
    ----------
    raw_segments : list[_RawSegment]
        All segments from Pass 1.
    candidates_by_label : dict[str, list[np.ndarray]]
        Gate 3 candidates after conflict resolution.
    vault : SpeakerVault
        Mutated in place by match_or_create().

    Returns
    -------
    list[TimelineSegment]
        Segments with global IDs and confidence flags assigned.
    """
    matched: list[TimelineSegment] = []
    # Track local label → global ID assignments for overlap_speakers resolution.
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

        # Record resolved global ID (excludes UNKNOWN and UNASSIGNED sentinels).
        if seg.speaker not in ("UNKNOWN", "UNASSIGNED", "OVERLAP"):
            label_to_global[seg.local_speaker_label] = seg.speaker

        matched.append(seg)

    # Resolve overlap_speakers: local labels → global IDs.
    # Must run after all segments are matched so label_to_global is complete.
    for raw in raw_segments:
        seg = raw.segment
        if raw.overlap_local_labels:
            seg.overlap_speakers = [
                label_to_global.get(lbl, lbl)  # fallback: keep local label if unresolved
                for lbl in raw.overlap_local_labels
            ]

    return matched


# ---------------------------------------------------------------------------
# Overlap detection helpers
# ---------------------------------------------------------------------------

def _detect_overlap(
    turn: "PySegment",
    local_speaker: str,
    diarization: "Annotation",
    overlap_timeline: "Timeline | None",
) -> tuple[bool, list[str]]:
    """
    Determine whether a turn contains concurrent speech and who else is active.

    Primary method: check overlap_timeline from diarization.get_overlap().
    Fallback: scan all other speaker turns for temporal intersection.
    The fallback is used when get_overlap() failed or returned None.

    Returns
    -------
    tuple[bool, list[str]]
        (is_overlap, overlap_local_labels)
        overlap_local_labels: local labels of co-active speakers.
    """
    is_overlap = False

    if overlap_timeline is not None:
        is_overlap = _segment_in_overlap(turn, overlap_timeline)
    else:
        # Fallback: check if any other speaker's turn intersects this one.
        for other_turn, _, other_speaker in diarization.itertracks(yield_label=True):
            if other_speaker == local_speaker:
                continue
            if other_turn.start < turn.end and other_turn.end > turn.start:
                is_overlap = True
                break

    overlap_local_labels: list[str] = []
    if is_overlap:
        overlap_local_labels = _find_coactive_speakers(turn, local_speaker, diarization)

    return is_overlap, overlap_local_labels


def _segment_in_overlap(turn: "PySegment", overlap_timeline: "Timeline") -> bool:
    """
    Returns True if the turn intersects any region in the overlap Timeline.
    """
    try:
        return len(overlap_timeline.crop(turn)) > 0
    except Exception:
        return False


def _find_coactive_speakers(
    turn: "PySegment",
    current_speaker: str,
    diarization: "Annotation",
) -> list[str]:
    """
    Find local speaker labels of all other speakers active during this turn.
    """
    coactive: list[str] = []
    for other_turn, _, other_speaker in diarization.itertracks(yield_label=True):
        if other_speaker == current_speaker:
            continue
        if other_turn.start < turn.end and other_turn.end > turn.start:
            if other_speaker not in coactive:
                coactive.append(other_speaker)
    return coactive


# ---------------------------------------------------------------------------
# Powerset posterior extraction
# ---------------------------------------------------------------------------

def _get_overlap_class_indices(pipeline) -> list[int]:
    """
    Identify which powerset class indices represent concurrent speech.

    INTERNAL API: pipeline._segmentation.model
    Not part of pyannote's public interface. Documented here and in
    the module docstring. Pinned to pyannote.audio >= 3.0.

    Powerset class layout for N max speakers:
        Index 0:     silence (0 speakers)
        Index 1..N:  single-speaker classes (one per speaker)
        Index N+1..: overlap combinations (2+ speakers active)

    Returns
    -------
    list[int]
        Overlap class indices. Empty list on any failure — caller
        degrades gracefully by setting concurrency_confidence = None.
    """
    try:
        model = pipeline._segmentation.model

        try:
            num_speakers = len(model.specifications.classes)
        except AttributeError:
            num_speakers = model.powerset.num_speakers

        try:
            num_classes = model.powerset.num_powerset_classes
        except AttributeError:
            num_classes = model.powerset.num_classes

        overlap_start = num_speakers + 1
        if overlap_start >= num_classes:
            logger.warning(
                "No overlap classes in powerset model "
                "(num_speakers=%d, num_classes=%d).",
                num_speakers, num_classes,
            )
            return []

        indices = list(range(overlap_start, num_classes))
        logger.debug(
            "Powerset: %d overlap classes %s (of %d total, %d speakers)",
            len(indices), indices, num_classes, num_speakers,
        )
        return indices

    except Exception as exc:
        logger.warning(
            "_get_overlap_class_indices failed: %s — "
            "concurrency_confidence will not be extracted.",
            exc,
        )
        return []


def _extract_posteriors(pipeline, audio_input: dict):
    """
    Extract raw powerset posterior probabilities for the chunk.

    INTERNAL API: pipeline._segmentation(audio_input)
    This is a second forward pass through the segmentation model only.
    The full pipeline() call has already run. This call takes ~0.1s
    additional overhead per chunk and is covered by torch.no_grad().

    Returns SlidingWindowFeature of shape (num_frames, num_classes),
    or None if the call fails. Failure is non-fatal — concurrency_confidence
    will be None for all segments in this chunk.
    """
    try:
        return pipeline._segmentation(audio_input)
    except Exception as exc:
        logger.warning(
            "_extract_posteriors: pipeline._segmentation failed: %s. "
            "concurrency_confidence will be None for this chunk.",
            exc,
        )
        return None


def _concurrency_confidence_for_segment(
    posteriors,
    overlap_class_indices: list[int],
    seg_start_local: float,
    seg_end_local: float,
) -> float | None:
    """
    Compute mean overlap class probability for a segment.

    For each frame in the segment, sums the probabilities of all
    overlap classes (2+ speakers). Averages across frames to produce
    a single scalar. This is the concurrency_confidence stored on the
    TimelineSegment.

    Parameters
    ----------
    posteriors : SlidingWindowFeature
        Output of pipeline._segmentation() — (num_frames, num_classes).
    overlap_class_indices : list[int]
        Class indices representing concurrent speech.
    seg_start_local, seg_end_local : float
        Segment boundaries in chunk-local time (seconds from chunk start).

    Returns
    -------
    float | None
        Mean per-frame overlap probability in [0, 1].
        None if the segment has no frames or cropping fails.
    """
    try:
        from pyannote.core import Segment as PySegment
        seg_obj = PySegment(seg_start_local, seg_end_local)

        # mode='loose' includes frames that partially overlap the boundary.
        frames = posteriors.crop(seg_obj, mode="loose")

        if frames is None or len(frames) == 0:
            return None

        # frames: (num_frames_in_segment, num_classes)
        overlap_probs = frames[:, overlap_class_indices]
        per_frame_overlap = overlap_probs.sum(axis=1)
        return float(np.clip(np.mean(per_frame_overlap), 0.0, 1.0))

    except Exception as exc:
        logger.debug(
            "concurrency_confidence failed [%.2fs, %.2fs]: %s",
            seg_start_local, seg_end_local, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def _load_chunk_audio(
    audio_path: Path,
    chunk_start: float,
    chunk_end: float,
    sample_rate: int,
) -> torch.Tensor:
    """
    Load a chunk of audio as a mono (1, num_samples) float32 tensor.

    Uses torchaudio's frame_offset / num_frames for efficient random access —
    only the requested samples are decoded, not the full file.

    The returned tensor is always mono. If the source file is stereo
    (which should not occur after Stage 3 normalization), channels are
    averaged. Sample rate mismatches trigger a warning and resample.
    """
    frame_offset = int(chunk_start * sample_rate)
    num_frames = int((chunk_end - chunk_start) * sample_rate)

    waveform, sr = torchaudio.load(
        str(audio_path),
        frame_offset=frame_offset,
        num_frames=num_frames,
    )

    if waveform.shape[0] > 1:
        logger.warning(
            "Chunk audio is %d-channel — expected mono from Stage 3. "
            "Averaging to mono.", waveform.shape[0],
        )
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        logger.warning(
            "Sample rate mismatch: expected %dHz, got %dHz. Resampling.",
            sample_rate, sr,
        )
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    return waveform


def _extract_segment_embedding(
    embed_inference: "Inference",
    chunk_waveform: torch.Tensor,
    sample_rate: int,
    seg_start_local: float,
    seg_end_local: float,
) -> np.ndarray | None:
    """
    Extract a 512-d ECAPA-TDNN embedding for one segment.

    Uses chunk-local times. Applies two zero-vector guards:
        1. Before inference: skip if waveform norm < ZERO_VECTOR_THRESHOLD.
           Prevents NaN embeddings from silent or corrupted frames.
        2. After inference: skip if embedding norm < ZERO_VECTOR_THRESHOLD.
           Catches pyannote extraction glitches that produce zero-vectors
           despite non-silent input.

    Handles both (512,) and (1, 512) output shapes from Inference.

    Returns None on any failure. Caller skips the segment entirely.
    """
    frame_start = int(seg_start_local * sample_rate)
    frame_end = min(int(seg_end_local * sample_rate), chunk_waveform.shape[1])

    if frame_end <= frame_start:
        return None

    seg_waveform = chunk_waveform[:, frame_start:frame_end]

    # Guard 1: waveform zero-vector.
    waveform_norm = float(torch.linalg.norm(seg_waveform))
    if waveform_norm < ZERO_VECTOR_THRESHOLD:
        logger.debug(
            "Segment [%.2fs, %.2fs]: near-zero waveform (norm=%.2e) — skipping.",
            seg_start_local, seg_end_local, waveform_norm,
        )
        return None

    try:
        audio_input = {"waveform": seg_waveform, "sample_rate": sample_rate}
        raw = embed_inference(audio_input)

        embedding: np.ndarray = np.asarray(raw, dtype=np.float32)
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)

        # Guard 2: embedding zero-vector.
        if np.linalg.norm(embedding) < ZERO_VECTOR_THRESHOLD:
            logger.warning(
                "Segment [%.2fs, %.2fs]: embed_inference returned near-zero embedding.",
                seg_start_local, seg_end_local,
            )
            return None

        return embedding

    except Exception as exc:
        logger.error(
            "Embedding extraction failed [%.2fs, %.2fs]: %s",
            seg_start_local, seg_end_local, exc,
        )
        return None


def _compute_rms(
    chunk_waveform: torch.Tensor,
    sample_rate: int,
    seg_start_local: float,
    seg_end_local: float,
) -> float | None:
    """
    Compute RMS energy for a segment's waveform slice.
    Returns None if the slice is empty or computation fails.
    """
    try:
        frame_start = int(seg_start_local * sample_rate)
        frame_end = min(int(seg_end_local * sample_rate), chunk_waveform.shape[1])
        if frame_end <= frame_start:
            return None
        seg_waveform = chunk_waveform[:, frame_start:frame_end]
        return float(torch.sqrt(torch.mean(seg_waveform ** 2)).item())
    except Exception as exc:
        logger.debug("RMS computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _generate_chunk_windows(
    audio_duration: float,
    chunk_size: float,
) -> list[tuple[float, float]]:
    """
    Generate overlapping (start, end) pairs covering the full audio.

    Adjacent chunks overlap by CHUNK_OVERLAP = 10.0 seconds.
    The last chunk extends exactly to audio_duration, which may make
    it shorter than chunk_size.

    Example for 12-minute audio, 5-minute chunks, 10s overlap:
        [(0.0, 300.0), (290.0, 590.0), (580.0, 720.0)]
    """
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < audio_duration:
        end = min(start + chunk_size, audio_duration)
        windows.append((start, end))
        if end >= audio_duration:
            break
        start = end - CHUNK_OVERLAP
    return windows


def _compute_core_window(
    chunk_start: float,
    chunk_end: float,
    is_first: bool,
    is_last: bool,
) -> tuple[float, float]:
    """
    Compute the core window in absolute file time.

    The core window excludes SEAM_HALF seconds from each non-edge
    boundary. Segments outside this window are discarded and covered
    by the adjacent chunk.

    Edge rules:
        First chunk — no left trim (no adjacent chunk to the left).
        Last chunk  — no right trim (no adjacent chunk to the right).
        Single chunk — no trim on either side.
    """
    core_start = chunk_start if is_first else chunk_start + SEAM_HALF
    core_end = chunk_end if is_last else chunk_end - SEAM_HALF
    return core_start, core_end