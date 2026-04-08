"""
src/diarization/vault/matching.py
===================================
Pure functions for cosine matching and confidence flag assignment.

Responsibility
--------------
Two stateless operations extracted from SpeakerVault:

    find_best_match()
        Scores a 512-d embedding against all existing vault centroids
        via cosine similarity. Returns the best match ID, its score,
        and the second-best score (used for ambiguity detection by the
        caller).

    apply_reid_flags()
        Applies MEDIUM_CONFIDENCE or LOW_CONFIDENCE to a segment based
        on its reid_score. HIGH confidence (>= 0.75) is silent — no
        flag added. Idempotent.

Design
------
Both functions are pure — they take explicit arguments and mutate
nothing except the segment passed to apply_reid_flags(). No vault
state is accessed here. This makes them independently testable and
lets metrics.py import from matching.py without risk of circular deps.

Ambiguity detection (FLAG_AMBIGUOUS_MATCH) is NOT performed here.
It requires both best_score and second_score together with the segment,
so it belongs inline in SpeakerVault._match_existing() where all three
are already in scope.

Contributors
------------
    Do not add vault state access here.
    Do not add gate logic here — that belongs in gates.py.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine

from ..segment import (
    FLAG_LOW_CONFIDENCE,
    FLAG_MEDIUM_CONFIDENCE,
    REID_HIGH_THRESHOLD,
    REID_MEDIUM_THRESHOLD,
    TimelineSegment,
)


def find_best_match(
    embedding: np.ndarray,
    anchors: dict[str, np.ndarray],
) -> tuple[str | None, float, float | None]:
    """
    Score a 512-d embedding against all vault centroids.

    Uses cosine similarity (1 - cosine distance). scipy.cosine()
    returns distance in [0, 1], so similarity = 1 - distance.

    Parameters
    ----------
    embedding : np.ndarray
        512-d ECAPA-TDNN embedding to match.
    anchors : dict[str, np.ndarray]
        Current vault centroids keyed by global speaker ID.
        Pass SpeakerVault.anchors directly.

    Returns
    -------
    tuple[str | None, float, float | None]
        (best_id, best_score, second_score)

        best_id      : global speaker ID with highest similarity,
                       or None if anchors is empty.
        best_score   : cosine similarity in [0, 1] for best_id,
                       or 0.0 if anchors is empty.
        second_score : cosine similarity for the second-best speaker,
                       or None if fewer than 2 anchors exist.
                       Used by the caller for AMBIGUOUS_MATCH detection.
    """
    if not anchors:
        return None, 0.0, None

    scores: list[tuple[str, float]] = sorted(
        (
            (gid, 1.0 - float(cosine(embedding, centroid)))
            for gid, centroid in anchors.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    best_id, best_score = scores[0]
    second_score = scores[1][1] if len(scores) >= 2 else None
    return best_id, best_score, second_score


def apply_reid_flags(segment: TimelineSegment) -> None:
    """
    Set confidence flags on a segment based on its reid_score.

    Flag assignment:
        reid_score >= REID_HIGH_THRESHOLD (0.75)  → no flag (HIGH, silent)
        reid_score >= REID_MEDIUM_THRESHOLD (0.50) → MEDIUM_CONFIDENCE
        reid_score <  REID_MEDIUM_THRESHOLD (0.50) → LOW_CONFIDENCE

    Idempotent — safe to call multiple times on the same segment.
    Does nothing if reid_score is None (e.g. OVERLAP or UNKNOWN
    segments where vault matching was not attempted).

    Parameters
    ----------
    segment : TimelineSegment
        Segment to flag. Mutated in place.
    """
    if segment.reid_score is None:
        return

    if segment.reid_score < REID_MEDIUM_THRESHOLD:
        if FLAG_LOW_CONFIDENCE not in segment.flags:
            segment.flags.append(FLAG_LOW_CONFIDENCE)
    elif segment.reid_score < REID_HIGH_THRESHOLD:
        if FLAG_MEDIUM_CONFIDENCE not in segment.flags:
            segment.flags.append(FLAG_MEDIUM_CONFIDENCE)