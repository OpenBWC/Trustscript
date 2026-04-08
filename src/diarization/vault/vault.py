"""
src/diarization/vault/vault.py
================================
TrustScript Phase 1 — Stage 5: SpeakerVault class.

Responsibility
--------------
Maintain global speaker identity across all diarization chunks.
Answers: "Is the voice in chunk 7 the same person from chunk 2?"

This module owns state and orchestration only. Pure computations
(cosine matching, spread metrics, quality block assembly) are
delegated to matching.py and metrics.py. Data structures live in
types.py. Gate logic lives in gates.py.

Centroid update correctness
----------------------------
Updates use a true incremental running mean:

    new_centroid = (old_centroid * n + embedding) / (n + 1)

A simple (old + new) / 2 is INCORRECT — it halves the weight of all
prior observations on every merge. The running mean preserves equal
weighting across all accepted embeddings regardless of merge count.

match_or_create() flow — two paths, cleanly separated
------------------------------------------------------
Existing anchor (best cosine score >= MATCH_THRESHOLD):
    1. Check for AMBIGUOUS_MATCH (top-2 scores within 0.05).
    2. Assign global ID and reid_score.
    3. Apply confidence flags via apply_reid_flags().
    4. Run vault gate (is_new_anchor=False).
    5a. Passes → _update_centroid().
        Apply PROVISIONAL flag if gate reason="PROVISIONAL".
    5b. Fails  → _record_rejection(), no centroid write.

New speaker (no match above threshold):
    1. Allocate speculative ID.
    2. Assign speaker=speculative_id, reid_score=1.0.
    3. Run vault gate (is_new_anchor=True, requires candidate_embeddings).
    4a. Passes → _create_entry().
        Apply PROVISIONAL flag if gate reason="PROVISIONAL".
    4b. Fails  → roll back ID, set speaker="UNKNOWN",
                 reid_score=None, _record_rejection().

_create_entry() and _update_centroid() are the only paths that write
vault state. Gate outcome is handled inline in each sub-path where
new-vs-existing context is explicit. This eliminates the double-write
risk of a shared gate-routing helper.

RMS handling
------------
rms is typed float | None = None throughout. An rms of 0.0 is valid
(digital silence) and must be stored. Truthiness checks (if rms:)
incorrectly treat 0.0 as falsy. All RMS writes use explicit None
comparison: if rms is not None.

Speaker ID scheme
-----------------
IDs are assigned in discovery order: TRUST_SPK_01, TRUST_SPK_02, ...
TRUST_SPK_01 is the most embedding-stable speaker in the first chunk.
Role assignment (officer, subject, bystander) is deferred to Phase 7.

Thread safety
-------------
Not thread-safe. engine.py processes chunks sequentially. Do not
share a vault instance across threads.

Contributors
------------
    Chunking and pyannote calls belong in engine.py.
    Gate logic belongs in gates.py.
    Pure computations belong in matching.py and metrics.py.
    Data structures belong in types.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine

from ..gates import passes_vault_gate
from ..segment import (
    AMBIGUITY_MARGIN,
    FLAG_AMBIGUOUS_MATCH,
    FLAG_HIGH_VARIANCE_SPEAKER,
    FLAG_OUTLIER_EMBEDDING,
    FLAG_PROVISIONAL,
    TimelineSegment,
)
from .matching import apply_reid_flags, find_best_match
from .metrics import build_quality_block, compute_spread, rejected_for_speaker
from .types import (
    HIGH_VARIANCE_THRESHOLD,
    MATCH_THRESHOLD,
    SPEAKER_ID_PREFIX,
    HistoryEntry,
    RejectedEntry,
    VaultEntry,
)

logger = logging.getLogger(__name__)


class SpeakerVault:
    """
    Global speaker identity vault for TrustScript Stage 5.

    Instantiate once per file. Pass to engine.py's chunking loop.
    After all chunks are processed, call get_vault_metadata() to
    assemble Stage 7 output.

        vault = SpeakerVault()
        for chunk in chunks:
            for segment, embedding, rms, candidates in chunk_results:
                vault.match_or_create(segment, embedding, rms, candidates)
        metadata = vault.get_vault_metadata()
    """

    def __init__(self) -> None:
        # Public — read by gates.py, engine.py, and stage5.py.
        self.anchors: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.durations: dict[str, float] = {}
        self.rms_values: dict[str, list[float]] = {}
        self.history: dict[str, list[dict]] = {}
        # Keyed by rejection reason string. Read by Phase 2.
        self.rejected_history: dict[str, list[RejectedEntry]] = {}

        # Internal.
        self._next_speaker_num: int = 1
        self._entries: dict[str, VaultEntry] = {}
        self._total_embeddings: int = 0
        self._accepted_embeddings: int = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def match_or_create(
        self,
        segment: TimelineSegment,
        embedding: np.ndarray,
        rms: float | None = None,
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
        rms : float | None
            RMS energy of this segment. Stored as metadata.
            0.0 is a valid value (digital silence) and is stored.
            None means RMS was not computed — not stored.
        candidate_embeddings : list[np.ndarray] | None
            All clean embeddings for this local speaker label in the
            current chunk. Required for Gate 3 when the speaker has
            no existing vault entry. Safe to pass for existing speakers
            — Gate 3 only runs when is_new_anchor=True.

        Returns
        -------
        str
            Global speaker ID assigned to the segment.
        """
        self._total_embeddings += 1

        best_id, best_score, second_score = find_best_match(embedding, self.anchors)
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
        rms: float | None = None,
    ) -> None:
        """
        Directly seed a new vault entry, bypassing gate checks.

        Called ONLY by stage5.py's vault initialization logic, after
        that code has applied its own stricter checks (SNR context,
        quality windows, stability). The gate bypass is intentional.

        Do NOT call from engine.py's chunking loop.
        Use match_or_create() for all runtime segments.

        Parameters
        ----------
        global_id : str
            Pre-assigned global ID, e.g. "TRUST_SPK_01".
        embedding : np.ndarray
            Seed embedding for this anchor.
        segment : TimelineSegment
            Originating segment (for history record).
        rms : float | None
            RMS energy. 0.0 is valid and stored. None = not computed.

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
        # Keep the speaker counter ahead of all seeded IDs.
        num = int(global_id.split("_")[-1])
        if num >= self._next_speaker_num:
            self._next_speaker_num = num + 1

        self._create_entry(global_id, embedding, segment, rms)
        logger.debug(
            "Vault: seeded anchor %s (segment %d)", global_id, segment.segment_id
        )

    def get_history_distances(self, global_id: str) -> list[float]:
        """
        Return cosine distances of all historical embeddings from the
        current centroid for the given speaker.

        Used by Gate 4 to compute the adaptive outlier threshold:
            threshold = mean_distance + (3 × std_dev)

        Returns
        -------
        list[float]
            Cosine distances in [0, 1] (0=identical, 1=orthogonal).
            Empty list if the speaker is not in the vault.
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
            Per-speaker spread metrics, duration, segment count, flags.
            No raw embeddings. Written to phase1.json.

        "vault_quality"
            Aggregate accepted/rejected counts and vault_purity_estimate.
            Written to both phase1.json and vault.json.

        "vault_detail"
            Full embedding history plus rejected embeddings per speaker.
            Written to vault.json only. Suppressed by --no-vault.
        """
        speakers = []
        for gid, entry in sorted(self._entries.items()):
            spread = compute_spread(entry)
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

        vault_quality = build_quality_block(
            self._total_embeddings,
            self._accepted_embeddings,
            self.rejected_history,
        )

        vault_detail = {
            gid: {
                "centroid": entry.centroid.tolist(),
                "embedding_spread": compute_spread(entry),
                "history": [
                    {
                        "segment_id": h["segment_id"],
                        "timestamp": h["timestamp"],
                        "embedding": h["embedding"].tolist(),
                    }
                    for h in entry.history
                ],
                "rejected_history": rejected_for_speaker(
                    gid, self.rejected_history
                ),
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
        rms: float | None,
        best_id: str,
        best_score: float,
        second_score: float | None,
        candidate_embeddings: list[np.ndarray] | None,
    ) -> None:
        """Existing-anchor path of match_or_create."""
        segment.speaker = best_id
        segment.reid_score = float(best_score)
        apply_reid_flags(segment)

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
            self._record_rejection(
                reason or "UNKNOWN_REJECTION", segment, embedding, best_id
            )
            logger.debug(
                "Segment %d: vault rejection (%s) for %s — centroid unchanged.",
                segment.segment_id, reason, best_id,
            )

    def _match_new(
        self,
        segment: TimelineSegment,
        embedding: np.ndarray,
        rms: float | None,
        candidate_embeddings: list[np.ndarray] | None,
    ) -> None:
        """New-speaker path of match_or_create."""
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
            # Roll back — this speaker is not vault-worthy.
            self._next_speaker_num -= 1
            segment.speaker = "UNKNOWN"
            segment.reid_score = None
            self._record_rejection(
                reason or "UNKNOWN_REJECTION", segment, embedding, speaker_id=None
            )
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
        rms: float | None,
    ) -> None:
        """
        Initialize a new vault entry for a previously unseen speaker.
        Only called after gate checks pass (or from seed_anchor).
        """
        entry = VaultEntry(
            speaker_id=global_id,
            centroid=embedding.copy(),
            count=1,
            total_duration=segment.duration_seconds,
            rms_values=[rms] if rms is not None else [],
            history=[
                HistoryEntry(
                    segment_id=segment.segment_id,
                    embedding=embedding.copy(),
                    timestamp=segment.start_seconds,
                )
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
        rms: float | None,
    ) -> None:
        """
        Merge an embedding into an existing centroid via running mean.

            new_centroid = (old_centroid * n + embedding) / (n + 1)

        Only called after gate checks pass.
        """
        entry = self._entries[global_id]
        n = entry.count
        entry.centroid = (entry.centroid * n + embedding) / (n + 1)
        entry.count += 1
        entry.total_duration += segment.duration_seconds
        if rms is not None:
            entry.rms_values.append(rms)
        entry.history.append(HistoryEntry(
            segment_id=segment.segment_id,
            embedding=embedding.copy(),
            timestamp=segment.start_seconds,
        ))
        self._sync_public_dicts(global_id, entry)
        self._accepted_embeddings += 1

        logger.debug(
            "Vault: updated %s — count=%d, segment=%d, reid=%.4f",
            global_id, entry.count, segment.segment_id,
            segment.reid_score or 0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_new_id(self) -> str:
        """Allocate the next TRUST_SPK_NN identifier."""
        gid = f"{SPEAKER_ID_PREFIX}_{self._next_speaker_num:02d}"
        self._next_speaker_num += 1
        return gid

    def _sync_public_dicts(self, global_id: str, entry: VaultEntry) -> None:
        """Keep the public-facing dicts consistent with the entry object."""
        self.anchors[global_id] = entry.centroid
        self.counts[global_id] = entry.count
        self.durations[global_id] = entry.total_duration
        self.rms_values[global_id] = entry.rms_values
        self.history[global_id] = entry.history

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
            RejectedEntry(
                segment_id=segment.segment_id,
                embedding=embedding.copy(),
                timestamp=segment.start_seconds,
                speaker_id=speaker_id,
            )
        )