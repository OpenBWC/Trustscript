"""
src/diarization/vault.py
=========================
TrustScript Phase 1 — Stage 5: Speaker Anchor Vault

Responsibility
--------------
Maintain global speaker identity across all diarization chunks.
Answers: "Is the voice in chunk 7 the same person from chunk 2?"

For each discovered speaker, the vault holds a 512-d centroid
embedding updated incrementally as new clean segments arrive.
Matching is cosine similarity against all live centroids.

Centroid update correctness
----------------------------
Updates use a true incremental running mean:

    new_centroid = (old_centroid * n + embedding) / (n + 1)

A simple (old + new) / 2 is INCORRECT — it halves the weight of
all prior observations on every merge, progressively losing early
anchor data. The running mean preserves equal weighting across all
accepted embeddings regardless of how many have been merged.

Vault gating
------------
Every embedding passes through passes_vault_gate() before any write.
A segment must pass all four gates. Rejected embeddings go to
rejected_history keyed by rejection reason. They are never discarded —
Phase 2 can consume them for mixed-signal analysis.

See gates.py for full gate documentation.

match_or_create() flow
-----------------------
The primary vault operation. Two paths, cleanly separated:

  Existing anchor path (best cosine score >= MATCH_THRESHOLD):
    1. Check ambiguity (top-2 within AMBIGUITY_MARGIN)
    2. Assign global ID and reid_score
    3. Apply reid confidence flags
    4. Run vault gate (is_new_anchor=False)
    5a. Gate passes → _update_centroid()
        Apply PROVISIONAL flag if gate returned reason="PROVISIONAL"
    5b. Gate fails  → _record_rejection(), no centroid write

  New speaker path (no match above threshold):
    1. Allocate speculative ID
    2. Assign speaker=speculative_id, reid_score=1.0
    3. Run vault gate (is_new_anchor=True, requires candidate_embeddings)
    4a. Gate passes → _create_entry()
        Apply PROVISIONAL flag if gate returned reason="PROVISIONAL"
    4b. Gate fails  → roll back ID, set speaker="UNKNOWN",
                      reid_score=None, _record_rejection()

_create_entry() and _update_centroid() are the only two paths that
write to vault state. There is no shared gate-routing helper — the
gate outcome is handled inline in match_or_create() where the
new-vs-existing context is explicit. This prevents the double-write
bug that a shared helper would introduce.

Speaker ID scheme
-----------------
IDs are assigned in discovery order: TRUST_SPK_01, TRUST_SPK_02, ...

TRUST_SPK_01 is the most stable speaker found in the first chunk —
not necessarily the officer or mic wearer. Role assignment is
deferred to the GroundTruth Interface in Phase 7.

Match threshold
---------------
MATCH_THRESHOLD = 0.65 cosine similarity.

Below this, an embedding is closer to "unknown new speaker" than any
existing anchor. Chosen to sit above the 0.05 ambiguity margin.
After collecting a BWC corpus, calibrate against known same-speaker /
different-speaker pairs and document the dataset here.

Thread safety
-------------
Not thread-safe. engine.py processes chunks sequentially — do not
share a vault instance across threads.

Contributors
------------
    Diarization and chunking logic belongs in engine.py.
    Pyannote calls belong in engine.py.
    Gate logic belongs in gates.py.
    Do not add per-chunk processing here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine

from ..gates import passes_vault_gate
from ..segment import (
    AMBIGUITY_MARGIN,
    FLAG_AMBIGUOUS_MATCH,
    FLAG_HIGH_VARIANCE_SPEAKER,
    FLAG_LOW_CONFIDENCE,
    FLAG_MEDIUM_CONFIDENCE,
    FLAG_OUTLIER_EMBEDDING,
    FLAG_PROVISIONAL,
    REID_HIGH_THRESHOLD,
    REID_MEDIUM_THRESHOLD,
    TimelineSegment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cosine similarity threshold for matching an embedding to an existing anchor.
#: Below this, the embedding is treated as a previously unseen speaker.
#: Sits above the 0.05 ambiguity margin to prevent near-misses from
#: collapsing two speakers into one identity.
#: Calibrate against a real BWC corpus — document dataset in a comment here.
MATCH_THRESHOLD: float = 0.65

#: Speaker ID prefix. All global IDs follow TRUST_SPK_NN format.
SPEAKER_ID_PREFIX: str = "TRUST_SPK"

#: Embedding spread std_dev above which a speaker is flagged
#: HIGH_VARIANCE_SPEAKER in vault metadata. May indicate identity confusion.
HIGH_VARIANCE_THRESHOLD: float = 0.20


# ---------------------------------------------------------------------------
# Internal per-speaker record
# ---------------------------------------------------------------------------

@dataclass
class _VaultEntry:
    """
    Internal per-speaker record. Not exposed outside this module.

    Attributes
    ----------
    speaker_id : str
        Global speaker ID, e.g. "TRUST_SPK_01".
    centroid : np.ndarray
        Current 512-d centroid. Updated on every accepted embedding
        via incremental running mean.
    count : int
        Number of embeddings merged into this centroid. Required for
        correct running mean: new = (old * n + emb) / (n + 1).
    total_duration : float
        Cumulative speech seconds assigned to this speaker.
    rms_values : list[float]
        RMS energy per accepted segment. Metadata only — drives no
        matching decisions. Available for downstream speaker prominence
        analysis.
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
    history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rejected embedding record
# ---------------------------------------------------------------------------

@dataclass
class _RejectedEntry:
    """
    One rejected embedding, stored in SpeakerVault.rejected_history
    keyed by rejection reason string.

    speaker_id is None when the rejection occurred before a global ID
    could be confirmed (e.g. Gate 3 failure on a new anchor candidate).
    """
    segment_id: int
    embedding: np.ndarray
    timestamp: float
    speaker_id: str | None


# ---------------------------------------------------------------------------
# SpeakerVault
# ---------------------------------------------------------------------------

class SpeakerVault:
    """
    Global speaker identity vault for TrustScript Stage 5.

    Maintains one centroid per discovered speaker. Matches incoming
    embeddings via cosine similarity and updates centroids using an
    incremental running mean. All centroid writes are gated.

    Usage
    -----
    Instantiate once per file. Pass to engine.py's chunking loop.
    After all chunks are processed, call get_vault_metadata() for
    Stage 7 output assembly.

        vault = SpeakerVault()
        for chunk in chunks:
            for segment, embedding, rms, candidates in chunk_results:
                vault.match_or_create(segment, embedding, rms, candidates)
        metadata = vault.get_vault_metadata()
    """

    def __init__(self) -> None:
        # Public — read by gates.py and engine.py.
        self.anchors: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.durations: dict[str, float] = {}
        self.rms_values: dict[str, list[float]] = {}
        self.history: dict[str, list[dict]] = {}
        # Keyed by rejection reason string, not by speaker.
        # engine.py and gates.py read this for audit trail.
        self.rejected_history: dict[str, list[_RejectedEntry]] = {}

        # Internal state.
        self._next_speaker_num: int = 1
        self._entries: dict[str, _VaultEntry] = {}

        # Vault-level counters for quality block.
        self._total_embeddings: int = 0
        self._accepted_embeddings: int = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def match_or_create(
        self,
        segment: TimelineSegment,
        embedding: np.ndarray,
        rms: float = 0.0,
        candidate_embeddings: list[np.ndarray] | None = None,
    ) -> str:
        """
        Match an embedding against existing anchors or create a new one.
        Mutates segment.speaker, segment.reid_score, and segment.flags.

        Parameters
        ----------
        segment : TimelineSegment
            Segment being matched. Mutated in place.
        embedding : np.ndarray
            512-d ECAPA-TDNN embedding for this segment.
        rms : float
            RMS energy of this segment. Stored as metadata.
        candidate_embeddings : list[np.ndarray] | None
            All clean embeddings for this local speaker label in the
            current chunk. Required for Gate 3 when the speaker has no
            existing vault entry. Safe to pass for existing speakers —
            Gate 3 only runs when is_new_anchor=True.

        Returns
        -------
        str
            The global speaker ID assigned to the segment.
        """
        self._total_embeddings += 1

        best_id, best_score, second_score = self._find_best_match(embedding)
        is_new_anchor = (best_id is None) or (best_score < MATCH_THRESHOLD)

        if not is_new_anchor:
            self._match_existing(
                segment, embedding, rms,
                best_id, best_score, second_score,
                candidate_embeddings,
            )
        else:
            self._match_new(segment, embedding, rms, candidate_embeddings)

        return segment.speaker

    def seed_anchor(
        self,
        global_id: str,
        embedding: np.ndarray,
        segment: TimelineSegment,
        rms: float = 0.0,
    ) -> None:
        """
        Directly seed a new vault entry, bypassing gate checks.

        Called ONLY by stage5.py's vault initialization logic, after
        that code has already applied its own stricter checks (SNR
        context, quality windows, stability). The gate bypass is
        intentional and documented — initialization has higher
        standards than the per-segment loop gates.

        Do NOT call from engine.py's main chunking loop.
        Use match_or_create() for all runtime segments.

        Parameters
        ----------
        global_id : str
            Pre-assigned global ID, e.g. "TRUST_SPK_01".
            Must follow TRUST_SPK_NN format.
        embedding : np.ndarray
            Seed embedding for this anchor.
        segment : TimelineSegment
            Originating segment (for history record).
        rms : float
            RMS energy of the seed segment.

        Raises
        ------
        ValueError
            If global_id already exists in the vault.
        """
        if global_id in self._entries:
            raise ValueError(
                f"seed_anchor() called for existing speaker {global_id}. "
                "Use match_or_create() to update existing anchors."
            )
        # Keep the speaker counter ahead of all seeded IDs so that
        # _assign_new_id() never collides with a seeded entry.
        num = int(global_id.split("_")[-1])
        if num >= self._next_speaker_num:
            self._next_speaker_num = num + 1

        self._create_entry(global_id, embedding, segment, rms)
        logger.debug("Vault: seeded anchor %s (segment %d)", global_id, segment.segment_id)

    def get_history_distances(self, global_id: str) -> list[float]:
        """
        Return cosine distances of all historical embeddings from the
        current centroid for the given speaker.

        Used by Gate 4 to compute the adaptive outlier threshold:
            threshold = mean_distance + (3 × std_dev)

        Returns
        -------
        list[float]
            Cosine distances (0=identical, 1=orthogonal).
            Empty list if speaker is not in the vault.
        """
        if global_id not in self._entries:
            return []
        centroid = self._entries[global_id].centroid
        return [
            float(cosine(h["embedding"], centroid))
            for h in self._entries[global_id].history
        ]

    def get_vault_metadata(self) -> dict[str, Any]:
        """
        Serialize vault state for Stage 7 output assembly.

        Returns a dict with three keys:

        "speakers"
            Per-speaker spread metrics, duration, segment count.
            No raw embedding arrays. Written to phase1.json.

        "vault_quality"
            Aggregate counts and vault_purity_estimate.
            Written to phase1.json and vault.json.

        "vault_detail"
            Full embedding history per speaker plus rejected embeddings
            per speaker. Written to vault.json only.
            Suppressed when --no-vault is passed to the CLI.
        """
        speakers = []
        for gid, entry in sorted(self._entries.items()):
            spread = self._compute_spread(entry)
            flags = []
            if (
                spread["std_dev"] is not None
                and spread["std_dev"] >= HIGH_VARIANCE_THRESHOLD
            ):
                flags.append(FLAG_HIGH_VARIANCE_SPEAKER)

            speakers.append({
                "speaker_id": gid,
                "suggested_role": None,
                "centroid_embedding": entry.centroid.tolist(),
                "embedding_spread": spread,
                "rms_mean": (
                    round(float(np.mean(entry.rms_values)), 6)
                    if entry.rms_values else None
                ),
                "total_duration_seconds": round(entry.total_duration, 3),
                "segment_count": entry.count,
                "flags": flags,
            })

        vault_quality = self._build_quality_block()

        vault_detail = {
            gid: {
                "centroid": entry.centroid.tolist(),
                "embedding_spread": self._compute_spread(entry),
                "history": [
                    {
                        "segment_id": h["segment_id"],
                        "timestamp": h["timestamp"],
                        "embedding": h["embedding"].tolist(),
                    }
                    for h in entry.history
                ],
                "rejected_history": self._rejected_for_speaker(gid),
            }
            for gid, entry in sorted(self._entries.items())
        }

        return {
            "speakers": speakers,
            "vault_quality": vault_quality,
            "vault_detail": vault_detail,
        }

    # ------------------------------------------------------------------
    # match_or_create sub-paths
    # ------------------------------------------------------------------

    def _match_existing(
        self,
        segment: TimelineSegment,
        embedding: np.ndarray,
        rms: float,
        best_id: str,
        best_score: float,
        second_score: float | None,
        candidate_embeddings: list[np.ndarray] | None,
    ) -> None:
        """Handle the existing-anchor path of match_or_create."""
        segment.speaker = best_id
        segment.reid_score = float(best_score)
        self._apply_reid_flags(segment)

        # Ambiguity check: top-2 scores within AMBIGUITY_MARGIN.
        if (
            second_score is not None
            and (best_score - second_score) <= AMBIGUITY_MARGIN
            and FLAG_AMBIGUOUS_MATCH not in segment.flags
        ):
            segment.flags.append(FLAG_AMBIGUOUS_MATCH)
            logger.debug(
                "Segment %d: AMBIGUOUS_MATCH — scores %.4f / %.4f for %s",
                segment.segment_id, best_score, second_score, best_id,
            )

        passed, reason = passes_vault_gate(
            segment=segment,
            embedding=embedding,
            vault=self,
            candidate_embeddings=candidate_embeddings,
            is_new_anchor=False,
        )

        if passed:
            if reason == "PROVISIONAL" and FLAG_PROVISIONAL not in segment.flags:
                segment.flags.append(FLAG_PROVISIONAL)
            self._update_centroid(best_id, embedding, segment, rms)
        else:
            if reason == "OUTLIER_EMBEDDING" and FLAG_OUTLIER_EMBEDDING not in segment.flags:
                segment.flags.append(FLAG_OUTLIER_EMBEDDING)
            self._record_rejection(reason or "UNKNOWN_REJECTION", segment, embedding, best_id)
            logger.debug(
                "Segment %d: vault rejection (%s) for %s — centroid not updated.",
                segment.segment_id, reason, best_id,
            )

    def _match_new(
        self,
        segment: TimelineSegment,
        embedding: np.ndarray,
        rms: float,
        candidate_embeddings: list[np.ndarray] | None,
    ) -> None:
        """Handle the new-speaker path of match_or_create."""
        speculative_id = self._assign_new_id()
        segment.speaker = speculative_id
        segment.reid_score = 1.0

        passed, reason = passes_vault_gate(
            segment=segment,
            embedding=embedding,
            vault=self,
            candidate_embeddings=candidate_embeddings,
            is_new_anchor=True,
        )

        if passed:
            if reason == "PROVISIONAL" and FLAG_PROVISIONAL not in segment.flags:
                segment.flags.append(FLAG_PROVISIONAL)
            self._create_entry(speculative_id, embedding, segment, rms)
            logger.debug(
                "Vault: new anchor %s (segment %d, t=%.2fs)",
                speculative_id, segment.segment_id, segment.start_seconds,
            )
        else:
            # Roll back the speculative ID — this speaker is not vault-worthy.
            self._next_speaker_num -= 1
            segment.speaker = "UNKNOWN"
            segment.reid_score = None
            self._record_rejection(reason or "UNKNOWN_REJECTION", segment, embedding, speaker_id=None)
            logger.debug(
                "Segment %d: new anchor rejected (%s) — ID rolled back.",
                segment.segment_id, reason,
            )

    # ------------------------------------------------------------------
    # Centroid write operations
    # ------------------------------------------------------------------

    def _create_entry(
        self,
        global_id: str,
        embedding: np.ndarray,
        segment: TimelineSegment,
        rms: float,
    ) -> None:
        """
        Initialize a new vault entry for a previously unseen speaker.
        Only called after all gate checks pass (or from seed_anchor).
        """
        entry = _VaultEntry(
            speaker_id=global_id,
            centroid=embedding.copy(),
            count=1,
            total_duration=segment.duration_seconds,
            rms_values=[rms] if rms else [],
            history=[
                {
                    "segment_id": segment.segment_id,
                    "embedding": embedding.copy(),
                    "timestamp": segment.start_seconds,
                }
            ],
        )
        self._entries[global_id] = entry
        self._sync_public_dicts(global_id, entry)
        self._accepted_embeddings += 1

    def _update_centroid(
        self,
        global_id: str,
        embedding: np.ndarray,
        segment: TimelineSegment,
        rms: float,
    ) -> None:
        """
        Merge an embedding into an existing centroid via running mean.

            new_centroid = (old_centroid * n + embedding) / (n + 1)

        Preserves equal weighting across all historical embeddings.
        Only called after gate checks pass.
        """
        entry = self._entries[global_id]
        n = entry.count
        entry.centroid = (entry.centroid * n + embedding) / (n + 1)
        entry.count += 1
        entry.total_duration += segment.duration_seconds
        if rms:
            entry.rms_values.append(rms)
        entry.history.append({
            "segment_id": segment.segment_id,
            "embedding": embedding.copy(),
            "timestamp": segment.start_seconds,
        })
        self._sync_public_dicts(global_id, entry)
        self._accepted_embeddings += 1

        logger.debug(
            "Vault: updated %s — count=%d, segment=%d, reid=%.4f",
            global_id, entry.count, segment.segment_id,
            segment.reid_score or 0.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_best_match(
        self, embedding: np.ndarray
    ) -> tuple[str | None, float, float | None]:
        """
        Score the embedding against all anchors via cosine similarity.

        Returns
        -------
        tuple[str | None, float, float | None]
            (best_id, best_score, second_score)
            best_id is None when the vault is empty.
            second_score is None when fewer than 2 anchors exist.
        """
        if not self.anchors:
            return None, 0.0, None

        scores: list[tuple[str, float]] = sorted(
            (
                (gid, 1.0 - float(cosine(embedding, centroid)))
                for gid, centroid in self.anchors.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        best_id, best_score = scores[0]
        second_score = scores[1][1] if len(scores) >= 2 else None
        return best_id, best_score, second_score

    def _assign_new_id(self) -> str:
        """Allocate the next TRUST_SPK_NN identifier."""
        gid = f"{SPEAKER_ID_PREFIX}_{self._next_speaker_num:02d}"
        self._next_speaker_num += 1
        return gid

    def _sync_public_dicts(self, global_id: str, entry: _VaultEntry) -> None:
        """Keep the public-facing dicts consistent with the entry object."""
        self.anchors[global_id] = entry.centroid
        self.counts[global_id] = entry.count
        self.durations[global_id] = entry.total_duration
        self.rms_values[global_id] = entry.rms_values
        self.history[global_id] = entry.history

    def _apply_reid_flags(self, segment: TimelineSegment) -> None:
        """
        Set MEDIUM_CONFIDENCE or LOW_CONFIDENCE based on reid_score.
        HIGH confidence (>= REID_HIGH_THRESHOLD) is silent — no flag.
        Idempotent — safe to call multiple times on the same segment.
        """
        if segment.reid_score is None:
            return
        if segment.reid_score < REID_MEDIUM_THRESHOLD:
            if FLAG_LOW_CONFIDENCE not in segment.flags:
                segment.flags.append(FLAG_LOW_CONFIDENCE)
        elif segment.reid_score < REID_HIGH_THRESHOLD:
            if FLAG_MEDIUM_CONFIDENCE not in segment.flags:
                segment.flags.append(FLAG_MEDIUM_CONFIDENCE)

    def _record_rejection(
        self,
        reason: str,
        segment: TimelineSegment,
        embedding: np.ndarray,
        speaker_id: str | None,
    ) -> None:
        """Append to rejected_history under the given reason key."""
        if reason not in self.rejected_history:
            self.rejected_history[reason] = []
        self.rejected_history[reason].append(
            _RejectedEntry(
                segment_id=segment.segment_id,
                embedding=embedding.copy(),
                timestamp=segment.start_seconds,
                speaker_id=speaker_id,
            )
        )

    def _rejected_for_speaker(self, global_id: str) -> dict[str, list[dict]]:
        """
        Filter rejected_history for entries belonging to this speaker,
        serialized by reason. Used by get_vault_metadata() for vault.json.
        Entries with speaker_id=None (Gate 3 failures on new anchors)
        are not attributed to any speaker and are excluded.
        """
        result: dict[str, list[dict]] = {}
        for reason, entries in self.rejected_history.items():
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

    def _build_quality_block(self) -> dict:
        """
        Build the vault_quality block for phase1.json output.

        GRACE_PERIOD_OUTLIER and OUTLIER_EMBEDDING are counted
        separately so calibration can distinguish early-segment
        anomalies from mature-history outliers.
        """
        total_rejected = sum(len(v) for v in self.rejected_history.values())
        return {
            "total_embeddings_extracted": self._total_embeddings,
            "accepted_into_vault": self._accepted_embeddings,
            "rejected_overlap": len(self.rejected_history.get("OVERLAP_REJECTED", [])),
            "rejected_too_short": len(self.rejected_history.get("TOO_SHORT", [])),
            "rejected_outlier": len(self.rejected_history.get("OUTLIER_EMBEDDING", [])),
            "rejected_grace_period_outlier": len(
                self.rejected_history.get("GRACE_PERIOD_OUTLIER", [])
            ),
            "rejected_insufficient_segments": len(
                self.rejected_history.get("INSUFFICIENT_SEGMENTS", [])
            ),
            "rejected_unstable_embeddings": len(
                self.rejected_history.get("UNSTABLE_EMBEDDINGS", [])
            ),
            "rejected_total": total_rejected,
            "vault_purity_estimate": round(
                self._accepted_embeddings / self._total_embeddings, 4
            ) if self._total_embeddings > 0 else 0.0,
        }

    @staticmethod
    def _compute_spread(entry: _VaultEntry) -> dict:
        """
        Compute embedding spread metrics for a vault entry.

        All values are None if fewer than 2 embeddings exist —
        spread is undefined for a single-embedding anchor.

        Returns dict with: mean_cosine_distance, max_cosine_distance, std_dev.
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