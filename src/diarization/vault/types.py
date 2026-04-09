"""
src/diarization/vault/types.py
================================
Vault-internal data structures and module-level constants.

Responsibility
--------------
Owns the two internal dataclasses used across the vault package
(VaultEntry, RejectedEntry) and the three tuning constants that
govern vault behaviour (MATCH_THRESHOLD, SPEAKER_ID_PREFIX,
HIGH_VARIANCE_THRESHOLD).

Separating these from the SpeakerVault class means:
    - matching.py and metrics.py can import types without importing
      the vault class, keeping the dependency graph acyclic.
    - Constants are in one place — not scattered across vault.py
      and the files that need to test against them.

No logic lives here. No imports from gates.py or segment.py —
this module has zero internal dependencies so it can be imported
anywhere in the package without risk of circular imports.

Contributors
------------
    Add new vault-internal dataclasses here as the vault grows.
    Do not add methods or computation — keep this purely structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cosine similarity threshold for matching an embedding to an existing anchor.
#: Below this, the embedding is treated as a previously unseen speaker.
#:
#: CALIBRATION NOTE — pyannote/speaker-diarization-community-1:
#: This model's ECAPA-TDNN embedding space has high intra-speaker cosine
#: distance under BWC conditions. Observed mean intra-speaker pairwise
#: distance on a clean single-speaker recording: 0.5157 → mean similarity
#: ~0.48. With MATCH_THRESHOLD = 0.65, segments of the same speaker
#: scoring below threshold are incorrectly routed to the new-anchor path,
#: causing speaker fragmentation (one person appearing as multiple TRUST_SPK
#: identities).
#:
#: Lowered to 0.45 based on observed embedding geometry. This captures
#: same-speaker variation while still providing separation from genuinely
#: different speakers. Calibrate against labelled BWC multi-speaker pairs
#: and document the calibration corpus here when done.
MATCH_THRESHOLD: float = 0.45

#: Prefix for all global speaker IDs: TRUST_SPK_01, TRUST_SPK_02, ...
SPEAKER_ID_PREFIX: str = "TRUST_SPK"

#: Embedding spread std_dev above which a speaker is flagged
#: HIGH_VARIANCE_SPEAKER in vault metadata. May indicate identity
#: confusion — two voices collapsed into one vault entry.
HIGH_VARIANCE_THRESHOLD: float = 0.20


# ---------------------------------------------------------------------------
# HistoryEntry
# ---------------------------------------------------------------------------

class HistoryEntry(TypedDict):
    """
    One accepted embedding in a speaker's vault history.

    Using TypedDict rather than a plain dict guarantees that every
    append to VaultEntry.history provides exactly these three keys
    with exactly these types. Missing or misspelled keys become type
    errors at the call site, not silent runtime bugs in downstream
    consumers (metrics.py, Phase 2 analysis, vault.json serialization).

    Fields
    ------
    segment_id : int
        Absolute segment index across the full file timeline.
    embedding : np.ndarray
        512-d ECAPA-TDNN embedding. Stored as a copy — mutations to
        the original array do not affect the history record.
    timestamp : float
        Absolute start time of the segment in the original file
        (seconds from t=0). Chunk offset has already been applied
        by the engine before this entry is created.
    """
    segment_id: int
    embedding: np.ndarray
    timestamp: float


# ---------------------------------------------------------------------------
# VaultEntry
# ---------------------------------------------------------------------------

@dataclass
class VaultEntry:
    """
    Per-speaker record held by SpeakerVault._entries.

    Not exposed outside the vault package. External consumers receive
    serialized dicts from get_vault_metadata(), not VaultEntry objects.

    Attributes
    ----------
    speaker_id : str
        Global speaker ID, e.g. "TRUST_SPK_01".
    centroid : np.ndarray
        Current 512-d centroid. Updated after every accepted embedding
        via incremental running mean:
            new = (old * n + embedding) / (n + 1)
    count : int
        Number of embeddings merged into this centroid. Required for
        correct incremental running mean.
    total_duration : float
        Cumulative speech seconds assigned to this speaker.
    rms_values : list[float]
        RMS energy per accepted segment. Metadata only — drives no
        matching decisions. None entries are never stored here; the
        list may be empty if no RMS values were supplied.
    history : list[dict]
        Every accepted embedding in order of occurrence.
        Each entry: {"segment_id": int, "embedding": np.ndarray,
                     "timestamp": float}
        Used post-run to compute spread metrics and by Phase 2
        for identity stability analysis.
    """
    speaker_id: str
    centroid: np.ndarray
    count: int = 1
    total_duration: float = 0.0
    rms_values: list[float] = field(default_factory=list)
    history: list[HistoryEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RejectedEntry
# ---------------------------------------------------------------------------

@dataclass
class RejectedEntry:
    """
    One rejected embedding stored in SpeakerVault.rejected_history.

    rejected_history is keyed by rejection reason string
    (e.g. "OVERLAP_REJECTED", "TOO_SHORT"). Each key maps to a list
    of RejectedEntry instances.

    Attributes
    ----------
    segment_id : int
        Segment that produced this embedding.
    embedding : np.ndarray
        512-d ECAPA-TDNN embedding. Stored for Phase 2 mixed-signal
        analysis and audit trail — never discarded.
    timestamp : float
        Absolute start time of the segment in the original file.
    speaker_id : str | None
        Global ID of the speaker the segment was assigned to before
        rejection, if one was confirmed. None when rejection occurred
        before a global ID could be assigned (e.g. Gate 3 failure on
        a new anchor candidate).
    """
    segment_id: int
    embedding: np.ndarray
    timestamp: float
    speaker_id: str | None