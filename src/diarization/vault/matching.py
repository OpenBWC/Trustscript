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

import logging
import math

import numpy as np
from scipy.spatial.distance import cosine

from ..segment import (
    FLAG_LOW_CONFIDENCE,
    FLAG_MEDIUM_CONFIDENCE,
    REID_HIGH_THRESHOLD,
    REID_MEDIUM_THRESHOLD,
    TimelineSegment,
)

# Instantiate the logger
logger = logging.getLogger(__name__)

 # Strip all existing confidence flags before re-applying.
# This ensures HIGH-confidence re-scores do not silently inherit
# LOW/MEDIUM flags from a prior assignment.
_CONFIDENCE_FLAGS = {FLAG_LOW_CONFIDENCE, FLAG_MEDIUM_CONFIDENCE}
   
    
def apply_reid_flags(segment: TimelineSegment) -> None:
    """
     Align confidence flags on a segment to its current reid_score.

    This is a state-aligner, not an appender. It first strips any
    existing confidence flags (LOW_CONFIDENCE, MEDIUM_CONFIDENCE),
    then applies exactly the correct flag for the current score.

    Stripping first is critical for Stage 6 re-scoring correctness.
    A segment initially scored at 0.45 (LOW_CONFIDENCE) that is
    re-scored at 0.85 after vault maturation must not retain
    LOW_CONFIDENCE in its flags — a reid_score/flag contradiction
    is an evidentiary failure in forensic output.

    Flag assignment after stripping:
        reid_score >= REID_HIGH_THRESHOLD (0.75)   → no flag (HIGH, silent)
        reid_score >= REID_MEDIUM_THRESHOLD (0.50) → MEDIUM_CONFIDENCE
        reid_score <  REID_MEDIUM_THRESHOLD (0.50) → LOW_CONFIDENCE

    Does nothing if reid_score is None (OVERLAP or UNKNOWN segments
    where vault matching was not attempted).

    Parameters
    ----------
    segment : TimelineSegment
        Segment to update. Mutated in place.
    """
    if segment.reid_score is None:
        return

    # 1. Strip existing confidence flags (O(1) lookup using the global set)
    segment.flags = [f for f in segment.flags if f not in _CONFIDENCE_FLAGS]

    # 2. Re-apply the correct flag
    if segment.reid_score < REID_MEDIUM_THRESHOLD:
        segment.flags.append(FLAG_LOW_CONFIDENCE)
    elif segment.reid_score < REID_HIGH_THRESHOLD:
        segment.flags.append(FLAG_MEDIUM_CONFIDENCE)

def find_best_match(
    embedding: np.ndarray,
    anchors: dict[str, np.ndarray],
) -> tuple[str | None, float, float | None]:
    """
    Score a 512-d embedding against all vault centroids.

    Uses cosine similarity (1 - cosine distance). scipy.cosine()
    returns distance in [0, 1], so similarity = 1 - distance.

    Zero-vector and NaN defence
    ----------------------------
    scipy.cosine(u, v) = 1 - (u·v) / (||u|| * ||v||). If either
    vector has zero magnitude — caused by a corrupted audio frame,
    upstream tensor collapse, or a pyannote extraction glitch — the
    denominator is zero and scipy returns NaN.

    A NaN score in the list does not raise during sorted() — Python's
    sort order becomes undefined and unpredictable, effectively
    randomising the speaker match silently. This is caught explicitly:
    any NaN or exception from cosine() is logged and the score for
    that pair is set to 0.0 (maximally dissimilar), ensuring the sort
    remains deterministic and the bad embedding does not win a match.

    Parameters
    ----------
    embedding : np.ndarray
        512-d ECAPA-TDNN embedding to match.
    anchors : dict[str, np.ndarray]
        Current vault centroids keyed by global speaker ID.

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

    scores: list[tuple[str, float]] = []
    for gid, centroid in anchors.items():
        try:
            dist = float(cosine(embedding, centroid))
            if math.isnan(dist):
                raise ValueError(
                    f"cosine() returned NaN for speaker {gid} — "
                    "embedding or centroid may be a zero-vector."
                )
            scores.append((gid, 1.0 - dist))
        except Exception as exc:  # noqa: BLE001
            # A corrupted embedding must not poison the sort or win a match.
            # Log at ERROR — this is unexpected and warrants investigation.
            # Treat the pair as maximally dissimilar (similarity = 0.0).
            logger.error(
                "find_best_match: cosine scoring failed for speaker %s — "
                "treating as dissimilar (score=0.0). Cause: %s",
                gid, exc,
            )
            scores.append((gid, 0.0))

    scores.sort(key=lambda x: x[1], reverse=True)

    best_id, best_score = scores[0]
    second_score = scores[1][1] if len(scores) >= 2 else None
    return best_id, best_score, second_score
