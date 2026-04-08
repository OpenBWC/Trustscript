"""
src/diarization/engine/overlap.py
====================================
Overlap detection helpers.

Responsibility
--------------
Determine whether a pyannote turn contains concurrent speech and
identify which other speakers are active during it.

Three functions, in call order:

    detect_overlap()
        Primary entry point. Checks the precomputed overlap_timeline
        from diarization.get_overlap() first. If that failed or
        returned None, falls back to scanning all other speaker turns
        for temporal intersection.

    segment_in_overlap()
        Tests whether a single pyannote Segment intersects any region
        in a precomputed overlap Timeline.

    find_coactive_speakers()
        Returns the local speaker labels of all other speakers whose
        turns intersect the given turn. Used to populate
        TimelineSegment.overlap_speakers (as local labels, later
        resolved to global IDs in passes._run_vault_matching).

No vault access, no audio I/O, no model calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyannote.core import Annotation, Segment as PySegment, Timeline

logger = logging.getLogger(__name__)


def detect_overlap(
    turn: "PySegment",
    local_speaker: str,
    diarization: "Annotation",
    overlap_timeline: "Timeline | None",
) -> tuple[bool, list[str]]:
    """
    Determine whether a turn contains concurrent speech.

    Primary method: check overlap_timeline from diarization.get_overlap().
    This is the fast path — precomputed by pyannote from the powerset
    output before the segment loop begins.

    Fallback: scan all other speaker turns for temporal intersection.
    Used when get_overlap() failed or returned None.

    Parameters
    ----------
    turn : PySegment
        The current speaker's turn in chunk-local time.
    local_speaker : str
        Local label for this turn (e.g. "SPEAKER_00").
    diarization : Annotation
        Full pyannote diarization result for the chunk.
    overlap_timeline : Timeline | None
        Precomputed overlap regions. None triggers the fallback.

    Returns
    -------
    tuple[bool, list[str]]
        (is_overlap, overlap_local_labels)
        overlap_local_labels: local speaker labels of co-active speakers.
        Empty list when is_overlap is False.
    """
    is_overlap = False

    if overlap_timeline is not None:
        is_overlap = segment_in_overlap(turn, overlap_timeline)
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
        overlap_local_labels = find_coactive_speakers(
            turn, local_speaker, diarization
        )

    return is_overlap, overlap_local_labels


def segment_in_overlap(
    turn: "PySegment",
    overlap_timeline: "Timeline",
) -> bool:
    """
    Test whether a turn intersects any region in the overlap Timeline.

    Uses pyannote's Timeline.crop() for fast interval intersection.
    Returns False on any exception — overlap detection errors are
    non-fatal; the segment continues to be processed as non-overlap.

    Parameters
    ----------
    turn : PySegment
        Turn in chunk-local time.
    overlap_timeline : Timeline
        Precomputed overlap regions from diarization.get_overlap().

    Returns
    -------
    bool
    """
    try:
        return len(overlap_timeline.crop(turn)) > 0
    except Exception:
        return False


def find_coactive_speakers(
    turn: "PySegment",
    current_speaker: str,
    diarization: "Annotation",
) -> list[str]:
    """
    Return local labels of all speakers active during this turn.

    Scans all other tracks in the diarization annotation and returns
    labels whose turns temporally intersect the given turn.

    Parameters
    ----------
    turn : PySegment
        The turn whose co-active speakers we want.
    current_speaker : str
        Local label of the speaker to exclude from results.
    diarization : Annotation
        Full pyannote diarization result for the chunk.

    Returns
    -------
    list[str]
        Local speaker labels of co-active speakers, deduplicated,
        in order of first appearance.
    """
    coactive: list[str] = []
    for other_turn, _, other_speaker in diarization.itertracks(yield_label=True):
        if other_speaker == current_speaker:
            continue
        if other_turn.start < turn.end and other_turn.end > turn.start:
            if other_speaker not in coactive:
                coactive.append(other_speaker)
    return coactive