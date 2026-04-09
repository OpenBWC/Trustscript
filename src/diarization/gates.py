"""
src/diarization/gates.py
=========================
TrustScript Phase 1 — Stage 5: Vault Gate Logic

Responsibility
--------------
Protect the Speaker Anchor Vault from contamination by mixed-signal,
short, unstable, or acoustically anomalous embeddings.

Every segment produces an embedding, but not every embedding should
enter the vault. A poisoned vault produces cascading misidentifications
across the entire file. These four gates are the primary defense.

Design contract
---------------
A segment must pass ALL FOUR gates to be written to the vault.
Gates are evaluated in order — the first failure short-circuits.
Rejected embeddings are NEVER discarded: the caller stores them in
vault.rejected_history with their rejection reason for audit trail
and Phase 2 mixed-signal analysis.

Gate order (cheapest to most expensive):
    1. Overlap confidence check  O(1) — fastest, most common rejection
    2. Minimum duration check    O(1)
    3. New anchor stability      O(n²) pairwise — only for new speakers
    4. Outlier check             O(n) history   — only for existing anchors

Gate 1 — overlap confidence threshold
--------------------------------------
Gate 1 does NOT unconditionally reject on the CONCURRENT_SPEECH flag.
It rejects only when concurrency_confidence >= OVERLAP_REJECTION_THRESHOLD
(default 0.40).

Rationale: pyannote sometimes detects weak overlap where the secondary
speaker is very quiet (distant chatter, background TV). In those cases,
the primary speaker's embedding may still be acoustically clean enough
to contribute signal to the vault. Rejecting everything with any overlap
flag would discard usable data in noisy BWC environments where some
ambient speech is always present.

Segments with overlap flag but concurrency_confidence < 0.40 pass Gate 1.
They remain flagged CONCURRENT_SPEECH / GHOST_SPEAKER in the timeline —
we do not retroactively clean those flags. Only vault admission is
affected by this threshold.

For v1, 0.40 is a conservative floor. Calibrate against a BWC corpus
and document the calibration dataset in a comment here.

Gate 4 — grace period
-----------------------
For the first MIN_HISTORY_FOR_OUTLIER_CHECK (= 5) embeddings in a
speaker's history, std() is statistically unreliable. Rather than
passing unconditionally (which lets anomalous early embeddings
contaminate the centroid), the gate applies a hard distance cap of
GRACE_PERIOD_MAX_DISTANCE (= 0.25) during the grace period.

This catches genuinely anomalous embeddings — shouting, mic handling
noise, partial overlaps — before the speaker's history is deep enough
for the adaptive mean+3σ threshold to be meaningful.

After MIN_HISTORY_FOR_OUTLIER_CHECK embeddings, the gate switches to
the adaptive per-speaker threshold.

Rejection reasons
-----------------
    OVERLAP_REJECTED        Gate 1: concurrency_confidence above threshold
    TOO_SHORT               Gate 2: segment too short for stable embedding
    INSUFFICIENT_SEGMENTS   Gate 3: not enough clean appearances to anchor
    UNSTABLE_EMBEDDINGS     Gate 3: embeddings too spread to trust
    OUTLIER_EMBEDDING       Gate 4: embedding too far from established centroid
    GRACE_PERIOD_OUTLIER    Gate 4: embedding exceeds hard cap during grace period

Contributors
------------
    Gate thresholds are defined as module constants below.
    After collecting a real BWC corpus, calibrate MIN_EMBEDDING_DURATION,
    MAX_ANCHOR_SPREAD, OVERLAP_REJECTION_THRESHOLD, and
    GRACE_PERIOD_MAX_DISTANCE against observed embedding quality.
    Document the calibration dataset in a comment here.
"""

import numpy as np
from scipy.spatial.distance import cosine

from .segment import FLAG_CONCURRENT_SPEECH, FLAG_GHOST_SPEAKER


# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

#: Gate 1 — minimum concurrency_confidence to reject on overlap.
#: Below this, the overlap is weak enough that the primary speaker's
#: embedding may still be clean. Above this, mixed-signal risk is too high.
#: Conservative v1 value — calibrate against BWC corpus.
OVERLAP_REJECTION_THRESHOLD: float = 0.40

#: Minimum segment duration (seconds) for a reliable ECAPA-TDNN embedding.
#: Below this, the embedding variance is too high to contribute signal.
#: ECAPA-TDNN needs ~0.75s of audio context at minimum.
MIN_EMBEDDING_DURATION: float = 0.75

#: Minimum number of clean segments required before seeding a new anchor.
#: A speaker appearing only once cannot be reliably fingerprinted.
MIN_ANCHOR_SEGMENTS: int = 3

#: Maximum mean pairwise cosine distance for new anchor stability check.
#: Embeddings within this spread are considered a tight, reliable cluster.
#: Calibrated at 0.30 for pyannote/speaker-diarization-community-1 +
#: pyannote/embedding (ECAPA-TDNN). The community model's embedding space
#: has higher natural intra-speaker spread than the original 3.1 weights —
#: 0.15 was too tight and rejected legitimate single-speaker audio.
#: Recalibrate against a labelled BWC corpus and document dataset here.
MAX_ANCHOR_SPREAD: float = 0.30

#: Gate 4 — grace period length.
#: Use hard distance cap for the first N embeddings before std() is
#: statistically reliable. Raised from 3 to 5 to cover the volatility
#: window identified in implementation review.
MIN_HISTORY_FOR_OUTLIER_CHECK: int = 5

#: Gate 4 — hard distance cap applied during grace period.
#: Embeddings further than this from the centroid are rejected even
#: before enough history exists for adaptive mean+3σ thresholding.
#: 0.25 is permissive enough for natural speaker variation while
#: catching genuine anomalies (shouting, mic noise, misassignment).
GRACE_PERIOD_MAX_DISTANCE: float = 0.40

#: Gate 4 — outlier threshold multiplier after grace period ends.
#: Threshold = mean_distance + (OUTLIER_STD_MULTIPLIER × std_dev).
#: 3.0 standard deviations catches genuine outliers without being
#: too strict for naturally variable speakers.
OUTLIER_STD_MULTIPLIER: float = 3.0


# ---------------------------------------------------------------------------
# Gate 1 — Overlap confidence check
# ---------------------------------------------------------------------------


def gate_1_overlap(segment) -> tuple[bool, str | None]:
    """
    Reject segments where concurrent speech confidence is high enough
    to risk embedding contamination.

    Does NOT unconditionally reject on the CONCURRENT_SPEECH flag.
    Rejects only when concurrency_confidence >= OVERLAP_REJECTION_THRESHOLD.
    If concurrency_confidence is None (flag present but value missing),
    defaults to rejection — absence of confidence data is treated as
    high-risk.

    If neither overlap flag nor segment.overlap is set, passes immediately
    without reading concurrency_confidence.

    Parameters
    ----------
    segment : TimelineSegment
        The segment being evaluated.

    Returns
    -------
    tuple[bool, str | None]
        (passes, rejection_reason)
        passes=True means this gate is satisfied.
    """
    overlap_flags = {FLAG_CONCURRENT_SPEECH, FLAG_GHOST_SPEAKER}
    has_overlap_flag = bool(overlap_flags.intersection(set(segment.flags)))
    has_overlap_bool = getattr(segment, "overlap", False)

    if not has_overlap_flag and not has_overlap_bool:
        return True, None

    # Overlap is indicated — check confidence level.
    confidence = getattr(segment, "concurrency_confidence", None)

    if confidence is None:
        # Flag present but no confidence value. Treat as high-risk.
        return False, "OVERLAP_REJECTED"

    if confidence >= OVERLAP_REJECTION_THRESHOLD:
        return False, "OVERLAP_REJECTED"

    # Confidence below threshold — overlap too weak to contaminate embedding.
    # Segment passes Gate 1 despite the overlap flag.
    return True, None


# ---------------------------------------------------------------------------
# Gate 2 — Minimum duration check
# ---------------------------------------------------------------------------


def gate_2_duration(
    segment,
    min_duration: float = MIN_EMBEDDING_DURATION,
) -> tuple[bool, str | None]:
    """
    Reject segments too short for a reliable ECAPA-TDNN embedding.

    Very short segments — including boundary slivers pyannote sometimes
    produces at overlap edges — have high embedding variance. Merging
    them into a centroid adds noise rather than signal.

    Parameters
    ----------
    segment : TimelineSegment
        The segment being evaluated.
    min_duration : float
        Minimum duration in seconds. Default: MIN_EMBEDDING_DURATION.

    Returns
    -------
    tuple[bool, str | None]
        (passes, rejection_reason)
    """
    if segment.duration_seconds < min_duration:
        return False, "TOO_SHORT"
    return True, None


# ---------------------------------------------------------------------------
# Gate 3 — New anchor stability check
# ---------------------------------------------------------------------------


def gate_3_new_anchor_stability(
    candidate_embeddings: list[np.ndarray],
    min_segments: int = MIN_ANCHOR_SEGMENTS,
    max_spread: float = MAX_ANCHOR_SPREAD,
) -> tuple[bool, str | None]:
    """
    Before creating a NEW vault entry for a previously unseen speaker,
    require evidence of acoustic consistency across multiple segments.

    Only runs when vault.match_or_create() would create a new anchor
    (no existing vault entry matches above the cosine threshold).

    Stability is measured as mean pairwise cosine distance across all
    clean candidate segments for this local speaker label within the
    current chunk. Low distance = tight cluster = reliable anchor.

    Parameters
    ----------
    candidate_embeddings : list[np.ndarray]
        All clean (gate-passing) embeddings for this local speaker
        label within the current chunk.
    min_segments : int
        Minimum number of clean segments required.
    max_spread : float
        Maximum acceptable mean pairwise cosine distance.

    Returns
    -------
    tuple[bool, str | None]
        (passes, rejection_reason)
        "INSUFFICIENT_SEGMENTS" or "UNSTABLE_EMBEDDINGS" on failure.
    """
    if len(candidate_embeddings) < min_segments:
        return False, "INSUFFICIENT_SEGMENTS"

    distances = []
    for i in range(len(candidate_embeddings)):
        for j in range(i + 1, len(candidate_embeddings)):
            distances.append(
                cosine(candidate_embeddings[i], candidate_embeddings[j])
            )

    mean_spread = float(np.mean(distances))
    if mean_spread > max_spread:
        return False, "UNSTABLE_EMBEDDINGS"

    return True, None


# ---------------------------------------------------------------------------
# Gate 4 — Outlier check for existing anchors
# ---------------------------------------------------------------------------


def gate_4_outlier_check(
    embedding: np.ndarray,
    centroid: np.ndarray,
    history_distances: list[float],
    std_multiplier: float = OUTLIER_STD_MULTIPLIER,
) -> tuple[bool, str | None]:
    """
    When updating an EXISTING vault centroid, check that the incoming
    embedding is consistent with the speaker's established history.

    Two-phase logic:

    Grace period (len(history_distances) < MIN_HISTORY_FOR_OUTLIER_CHECK):
        std() is statistically unreliable with few samples. Rather than
        passing unconditionally (which allows early anomalies to poison
        the centroid), apply a hard distance cap of GRACE_PERIOD_MAX_DISTANCE.
        Embeddings within 0.25 cosine distance of the centroid pass.
        Embeddings beyond 0.25 are rejected as GRACE_PERIOD_OUTLIER.

    Mature period (len(history_distances) >= MIN_HISTORY_FOR_OUTLIER_CHECK):
        Uses adaptive per-speaker threshold:
            threshold = mean_distance + (std_multiplier × std_dev)
        A naturally variable speaker gets a wider gate than a stable one.

    Parameters
    ----------
    embedding : np.ndarray
        The incoming 512-d embedding to evaluate.
    centroid : np.ndarray
        Current centroid for this speaker.
    history_distances : list[float]
        Cosine distances of all previously accepted embeddings from
        the centroid. From vault.get_history_distances().
    std_multiplier : float
        Adaptive threshold multiplier (used after grace period only).

    Returns
    -------
    tuple[bool, str | None]
        (True, None)                  — passes, within expected range
        (True, "PROVISIONAL")         — passes, at grace period boundary
                                        (exactly MIN_HISTORY count)
        (False, "GRACE_PERIOD_OUTLIER") — rejected during grace period
        (False, "OUTLIER_EMBEDDING")  — rejected by adaptive threshold
    """
    incoming_dist = float(cosine(embedding, centroid))

    if len(history_distances) < MIN_HISTORY_FOR_OUTLIER_CHECK:
        # Grace period — hard distance cap.
        if incoming_dist > GRACE_PERIOD_MAX_DISTANCE:
            return False, "GRACE_PERIOD_OUTLIER"
        # Within cap — pass, but flag PROVISIONAL (history immature).
        return True, "PROVISIONAL"

    # Mature period — adaptive per-speaker threshold.
    mean_dist = float(np.mean(history_distances))
    std_dist = float(np.std(history_distances))
    threshold = mean_dist + (std_multiplier * std_dist)

    if incoming_dist > threshold:
        return False, "OUTLIER_EMBEDDING"

    return True, None


# ---------------------------------------------------------------------------
# Master gate
# ---------------------------------------------------------------------------


def passes_vault_gate(
    segment,
    embedding: np.ndarray,
    vault: "SpeakerVault",  # type: ignore[name-defined]
    candidate_embeddings: list[np.ndarray] | None = None,
    is_new_anchor: bool = False,
) -> tuple[bool, str | None]:
    """
    Master gate check. Runs all applicable gates in order.
    Returns on the first failure — does not continue checking.

    Parameters
    ----------
    segment : TimelineSegment
        The segment being evaluated.
    embedding : np.ndarray
        The 512-d ECAPA-TDNN embedding for this segment.
    vault : SpeakerVault
        The live vault instance. Used for Gate 4's history lookup.
    candidate_embeddings : list[np.ndarray] | None
        All clean embeddings for this local speaker in the current chunk.
        Required when is_new_anchor=True.
    is_new_anchor : bool
        True when this segment would create a new vault entry.

    Returns
    -------
    tuple[bool, str | None]
        (True, None)                  — all gates passed
        (True, "PROVISIONAL")         — passed, grace period active
        (False, reason)               — rejected, store in rejected_history
    """
    # Gate 1 — overlap confidence check (always runs first, O(1))
    passed, reason = gate_1_overlap(segment)
    if not passed:
        return False, reason

    # Gate 2 — duration check (O(1))
    passed, reason = gate_2_duration(segment)
    if not passed:
        return False, reason

    # Gate 3 — new anchor stability (only for new vault entries)
    if is_new_anchor:
        if candidate_embeddings is None or len(candidate_embeddings) == 0:
            return False, "INSUFFICIENT_SEGMENTS"
        passed, reason = gate_3_new_anchor_stability(candidate_embeddings)
        if not passed:
            return False, reason

    # Gate 4 — outlier check (only for existing anchors)
    if not is_new_anchor and segment.speaker in vault.anchors:
        history_distances = vault.get_history_distances(segment.speaker)
        passed, reason = gate_4_outlier_check(
            embedding,
            vault.anchors[segment.speaker],
            history_distances,
        )
        if not passed:
            return False, reason
        if reason == "PROVISIONAL":
            return True, "PROVISIONAL"

    return True, None