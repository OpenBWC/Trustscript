"""
src/diarization/segment.py
===========================
TrustScript Phase 1 — Stage 5: Timeline Segment Data Model

Responsibility
--------------
The TimelineSegment dataclass is the atomic unit of the diarization
timeline. Every segment produced by the windowed diarization loop is
represented as one TimelineSegment instance.

This module owns only the data model and its serialization.
No diarization logic, no vault logic, no pyannote calls live here.

Signal Time contract
--------------------
start_seconds and end_seconds are derived from pyannote's output,
which already operates in the original file's time space. Unlike
Stage 2's frame-based analysis, pyannote returns segment boundaries
directly in seconds relative to the start of the audio passed to it.

For a chunk starting at offset 300.0s, pyannote returns a segment
at t=4.2s within that chunk. The engine is responsible for adding
the chunk offset before storing on TimelineSegment:

    segment.start_seconds = chunk_start + pyannote_segment.start

This means TimelineSegment always carries absolute file-level times,
not chunk-relative times.

Field lifecycle: transient vs. persisted
-----------------------------------------
Most fields are written to disk via to_dict() and round-trip through
from_dict(). One field is intentionally transient:

    local_speaker_label (str, default "")
        pyannote's chunk-local label (e.g. "SPEAKER_00"). Used by the
        engine during the Stage 5 loop to map local labels to global
        vault IDs. Has no meaning after the chunk is processed.
        Omitted from to_dict() — not written to any output file.
        Defaults to "" so from_dict() and any deserialization path
        can construct a valid TimelineSegment without supplying it.

Flag vocabulary
---------------
Flags are string constants that accumulate on a segment as it moves
through the pipeline. Multiple flags can coexist on one segment.

Stage 5 flags (set by engine.py / vault.py):
    PROVISIONAL             Vault was immature at assignment time.
                            Candidate for Stage 6 re-scoring.
    AMBIGUOUS_MATCH         Top-2 vault cosine scores within 0.05.
                            Speaker identity uncertain.
    OUTLIER_EMBEDDING       Embedding rejected by Gate 4 adaptive threshold.
                            Segment assigned but embedding not merged.
    LOW_ANCHOR_CONFIDENCE   First chunk was too chaotic to seed vault.
                            All first-chunk segments carry this.
    CONCURRENT_SPEECH       Overlap detected (concurrency_confidence ≥ 0.65).
    GHOST_SPEAKER           Weak overlap (concurrency_confidence < 0.65).
    MEDIUM_CONFIDENCE       reid_score 0.50–0.74.
    LOW_CONFIDENCE          reid_score < 0.50.
    HIGH_VARIANCE_SPEAKER   Speaker's embedding spread exceeds std_dev threshold.
                            May indicate identity confusion in the vault.

Stage 6 flags (set by stage6.py / rescoring.py):
    RESCORED                PROVISIONAL segment successfully re-scored.
    RESCORED_AMBIGUOUS      Re-scored but still ambiguous after mature vault.

Contributors
------------
    Add new flag string constants to this module when new flag types
    are introduced. Keep flag names UPPER_SNAKE_CASE.
    Do not add computation logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Flag constants
# ---------------------------------------------------------------------------

# Stage 5 flags
FLAG_PROVISIONAL            = "PROVISIONAL"
FLAG_AMBIGUOUS_MATCH        = "AMBIGUOUS_MATCH"
FLAG_OUTLIER_EMBEDDING      = "OUTLIER_EMBEDDING"
FLAG_LOW_ANCHOR_CONFIDENCE  = "LOW_ANCHOR_CONFIDENCE"
FLAG_CONCURRENT_SPEECH      = "CONCURRENT_SPEECH"
FLAG_GHOST_SPEAKER          = "GHOST_SPEAKER"
FLAG_MEDIUM_CONFIDENCE      = "MEDIUM_CONFIDENCE"
FLAG_LOW_CONFIDENCE         = "LOW_CONFIDENCE"
FLAG_HIGH_VARIANCE_SPEAKER  = "HIGH_VARIANCE_SPEAKER"

# Stage 6 flags
FLAG_RESCORED               = "RESCORED"
FLAG_RESCORED_AMBIGUOUS     = "RESCORED_AMBIGUOUS"


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

#: reid_score at or above this → HIGH confidence (no flag).
REID_HIGH_THRESHOLD: float = 0.75

#: reid_score between this and HIGH → MEDIUM_CONFIDENCE flag.
REID_MEDIUM_THRESHOLD: float = 0.50

#: Top-2 cosine scores within this margin → AMBIGUOUS_MATCH flag.
AMBIGUITY_MARGIN: float = 0.05

#: concurrency_confidence at or above this → confirmed overlap.
CONCURRENCY_CONFIRMED_THRESHOLD: float = 0.65


# ---------------------------------------------------------------------------
# TimelineSegment
# ---------------------------------------------------------------------------


@dataclass
class TimelineSegment:
    """
    One segment of the diarization timeline.

    Represents a contiguous span of audio assigned to a single speaker
    (or marked as overlapping). Produced by engine.py during the
    windowed diarization loop and consumed by vault.py, stage5.py,
    stage6.py, and ultimately Stage 7's output assembler.

    Attributes
    ----------
    segment_id : int
        Zero-based sequential index across the full file timeline.
        Assigned by the engine as segments are collected.
    start_seconds : float
        Absolute start time in the original file (seconds from t=0).
        Chunk offset is added by the engine before this is set.
    end_seconds : float
        Absolute end time in the original file.
    duration_seconds : float
        end_seconds - start_seconds.
    speaker : str
        Global speaker ID assigned by the vault: "TRUST_SPK_01", etc.
        "OVERLAP" when the segment is concurrent speech.
        "UNKNOWN" when vault matching failed with no assignment.
    local_speaker_label : str
        pyannote's local label within the chunk: "SPEAKER_00", etc.
        Transient — used only during the Stage 5 chunking loop to map
        local labels to global vault IDs. Omitted from to_dict() and
        not written to any output file. Defaults to "" so that
        from_dict() and any deserialization path can construct a valid
        TimelineSegment without supplying it.
    reid_score : float | None
        Cosine similarity between this segment's embedding and the
        vault centroid it was matched to.
        None for OVERLAP segments (no vault match attempted).
        1.0 for newly created vault anchors.
    reid_score_original : float | None
        Original reid_score before Stage 6 re-scoring.
        None until Stage 6 runs. If set, indicates this segment
        was re-scored and the original uncertainty is preserved.
    overlap : bool
        True when concurrent speech was detected in this segment.
    overlap_speakers : list[str]
        Global speaker IDs involved in the overlap.
        Empty when overlap is False.
    concurrency_confidence : float | None
        Raw softmax probability from pyannote's powerset posteriors
        that this segment contains concurrent speech.
        None for non-overlap segments.
    flags : list[str]
        Accumulated quality and uncertainty flags.
        See module-level flag constants for vocabulary.
    chunk_index : int
        Which diarization chunk this segment originated from.
        Used for debugging and provenance — not in output JSON.
    """
    segment_id: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    speaker: str
    reid_score: float | None
    # Transient — engine-internal only, omitted from to_dict().
    # Default="" means from_dict() and deserialization never need to supply it.
    local_speaker_label: str = ""
    reid_score_original: float | None = None
    overlap: bool = False
    overlap_speakers: list[str] = field(default_factory=list)
    concurrency_confidence: float | None = None
    flags: list[str] = field(default_factory=list)
    chunk_index: int = 0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_provisional(self) -> bool:
        return FLAG_PROVISIONAL in self.flags

    @property
    def is_ambiguous(self) -> bool:
        return FLAG_AMBIGUOUS_MATCH in self.flags

    @property
    def is_overlap(self) -> bool:
        return self.overlap

    @property
    def is_ghost_speaker(self) -> bool:
        return FLAG_GHOST_SPEAKER in self.flags

    @property
    def needs_review(self) -> bool:
        """True if this segment should surface in the human review queue."""
        review_flags = {
            FLAG_PROVISIONAL,
            FLAG_AMBIGUOUS_MATCH,
            FLAG_LOW_CONFIDENCE,
            FLAG_LOW_ANCHOR_CONFIDENCE,
            FLAG_RESCORED_AMBIGUOUS,
        }
        return bool(review_flags.intersection(self.flags))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize to the timeline segment format for JSON output.

        local_speaker_label is intentionally excluded — it is a
        transient engine-internal field with no meaning after Stage 5.

        Used by Stage 5 when writing the intermediate
        {incident_id}_timeline.json and by Stage 7 when assembling
        the final {incident_id}_phase1.json.
        """
        return {
            "segment_id": self.segment_id,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "speaker": self.speaker,
            "reid_score": (
                round(self.reid_score, 4)
                if self.reid_score is not None else None
            ),
            "reid_score_original": (
                round(self.reid_score_original, 4)
                if self.reid_score_original is not None else None
            ),
            "overlap": self.overlap,
            "overlap_speakers": self.overlap_speakers,
            "concurrency_confidence": (
                round(self.concurrency_confidence, 4)
                if self.concurrency_confidence is not None else None
            ),
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineSegment":
        """
        Deserialize from a timeline segment dict.

        local_speaker_label is not present in serialized data (omitted
        by to_dict). It is not supplied here — the field default of ""
        is used. Stage 6 operates entirely on global speaker IDs and
        never needs the original chunk-local label.

        Used by Stage 6 when reading the Stage 5 intermediate timeline
        file from disk for re-scoring.
        """
        return cls(
            segment_id=data["segment_id"],
            start_seconds=data["start_seconds"],
            end_seconds=data["end_seconds"],
            duration_seconds=data["duration_seconds"],
            speaker=data["speaker"],
            reid_score=data.get("reid_score"),
            # local_speaker_label intentionally not supplied — defaults to "".
            reid_score_original=data.get("reid_score_original"),
            overlap=data.get("overlap", False),
            overlap_speakers=data.get("overlap_speakers", []),
            concurrency_confidence=data.get("concurrency_confidence"),
            flags=data.get("flags", []),
            chunk_index=data.get("chunk_index", 0),
        )