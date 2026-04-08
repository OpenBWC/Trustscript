"""
src/diarization/vault/metrics.py
==================================
Pure functions for vault spread metrics and output serialization.

Responsibility
--------------
Three stateless functions extracted from SpeakerVault:

    compute_spread()
        Computes mean/max cosine distance and std_dev for a speaker's
        embedding history relative to their current centroid.
        These are the primary signals for identity confusion detection
        in Phase 5 Fusion.

    build_quality_block()
        Assembles the vault_quality dict written to phase1.json.
        Counts accepted and rejected embeddings by rejection reason.
        Counts GRACE_PERIOD_OUTLIER and OUTLIER_EMBEDDING separately
        so calibration can distinguish early-segment anomalies from
        mature-history outliers.

    rejected_for_speaker()
        Filters the flat rejected_history dict (keyed by reason) for
        entries belonging to a specific speaker. Serializes to the
        per-speaker rejected_history block in vault.json.

Design
------
All three functions are pure — they take data structures as arguments
and return new dicts. No vault state is accessed directly. This means:
    - They can be tested against synthetic VaultEntry / RejectedEntry
      objects without instantiating a full SpeakerVault.
    - get_vault_metadata() in vault.py becomes a thin assembly function.

Rejection reason strings are used as literals here. They originate
in gates.py and are preserved exactly as returned by the gate
functions. A future refactor could centralise these as constants in
types.py — for now, literals avoid a cross-package dependency from
metrics.py back to gates.py.

Contributors
------------
    Do not add vault state access here.
    Do not add matching logic here — that belongs in matching.py.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine

from .types import RejectedEntry, VaultEntry


def compute_spread(entry: VaultEntry) -> dict:
    """
    Compute embedding spread metrics for a vault entry.

    Spread quantifies how tightly clustered a speaker's accepted
    embeddings are around their current centroid:

        Low spread  → tight cluster → reliable, stable identity
        High spread → wide cluster  → possible identity confusion,
                                      vocal stress, or misassignment

    All values are None when fewer than 2 embeddings exist in history.
    Spread is undefined for a single-embedding anchor — the caller
    should not flag HIGH_VARIANCE_SPEAKER on None std_dev.

    Parameters
    ----------
    entry : VaultEntry
        The speaker's vault record.

    Returns
    -------
    dict
        {
            "mean_cosine_distance": float | None,
            "max_cosine_distance":  float | None,
            "std_dev":              float | None,
        }
        All values rounded to 6 decimal places when present.
    """
    if len(entry.history) < 2:
        return {
            "mean_cosine_distance": None,
            "max_cosine_distance": None,
            "std_dev": None,
        }

    centroid = entry.centroid
    distances = [
        float(cosine(h["embedding"], centroid))
        for h in entry.history
    ]
    return {
        "mean_cosine_distance": round(float(np.mean(distances)), 6),
        "max_cosine_distance": round(float(np.max(distances)), 6),
        "std_dev": round(float(np.std(distances)), 6),
    }


def build_quality_block(
    total_embeddings: int,
    accepted_embeddings: int,
    rejected_history: dict[str, list[RejectedEntry]],
) -> dict:
    """
    Assemble the vault_quality block written to phase1.json.

    vault_purity_estimate = accepted / total. A file with purity below
    0.5 means more than half the audio was too noisy or ambiguous for
    reliable diarization — important context for every downstream phase.

    GRACE_PERIOD_OUTLIER and OUTLIER_EMBEDDING are counted separately.
    GRACE_PERIOD_OUTLIER = anomalies caught during the first N embeddings
    before std() is statistically reliable (see gates.py Gate 4).
    OUTLIER_EMBEDDING = anomalies caught by the mature adaptive threshold.
    This distinction enables post-run calibration of the grace period cap.

    Parameters
    ----------
    total_embeddings : int
        Total embeddings extracted across all chunks and all speakers.
    accepted_embeddings : int
        Embeddings that passed all vault gates and updated a centroid.
    rejected_history : dict[str, list[RejectedEntry]]
        The vault's flat rejected_history, keyed by rejection reason.

    Returns
    -------
    dict
        vault_quality block ready for JSON serialization.
    """
    total_rejected = sum(len(v) for v in rejected_history.values())
    purity = (
        round(accepted_embeddings / total_embeddings, 4)
        if total_embeddings > 0
        else 0.0
    )
    return {
        "total_embeddings_extracted": total_embeddings,
        "accepted_into_vault": accepted_embeddings,
        "rejected_overlap": len(rejected_history.get("OVERLAP_REJECTED", [])),
        "rejected_too_short": len(rejected_history.get("TOO_SHORT", [])),
        "rejected_outlier": len(rejected_history.get("OUTLIER_EMBEDDING", [])),
        "rejected_grace_period_outlier": len(
            rejected_history.get("GRACE_PERIOD_OUTLIER", [])
        ),
        "rejected_insufficient_segments": len(
            rejected_history.get("INSUFFICIENT_SEGMENTS", [])
        ),
        "rejected_unstable_embeddings": len(
            rejected_history.get("UNSTABLE_EMBEDDINGS", [])
        ),
        "rejected_total": total_rejected,
        "vault_purity_estimate": purity,
    }


def rejected_for_speaker(
    global_id: str,
    rejected_history: dict[str, list[RejectedEntry]],
) -> dict[str, list[dict]]:
    """
    Filter the flat rejected_history for entries belonging to one speaker.

    The vault stores rejected embeddings in a flat structure keyed by
    rejection reason, not by speaker. This function reconstructs the
    per-speaker view used in vault.json output:

        "rejected_history": {
            "OVERLAP_REJECTED": [{segment_id, timestamp, embedding}, ...],
            "OUTLIER_EMBEDDING": [{...}],
        }

    Entries with speaker_id=None (Gate 3 failures on new anchor
    candidates before any global ID was confirmed) are intentionally
    excluded. They are not attributable to any speaker and would be
    misleading in a per-speaker block.

    Parameters
    ----------
    global_id : str
        The global speaker ID to filter for (e.g. "TRUST_SPK_01").
    rejected_history : dict[str, list[RejectedEntry]]
        The vault's flat rejected_history.

    Returns
    -------
    dict[str, list[dict]]
        Rejection reason → list of serialized entries.
        Empty reasons are omitted from the output.
    """
    result: dict[str, list[dict]] = {}
    for reason, entries in rejected_history.items():
        matching = [
            {
                "segment_id": e.segment_id,
                "timestamp": e.timestamp,
                "embedding": e.embedding.tolist(),
            }
            for e in entries
            if e.speaker_id == global_id
        ]
        if matching:
            result[reason] = matching
    return result