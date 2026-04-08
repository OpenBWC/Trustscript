"""
src/diarization/engine/candidates.py
=======================================
Gate 3 candidate pool preparation and local conflict resolution.

Responsibility
--------------
Two functions that must run after Pass 1 completes and before
vault.match_or_create() is called for any segment:

    build_candidates_by_label()
        Collects clean (non-overlap) embeddings per local speaker label.
        These are the Gate 3 evidence pools. Gate 3 requires N >= 3
        clean segments with tight pairwise spread before creating a
        new vault anchor.

    resolve_local_conflicts()
        Detects when two chunk-local labels both score above
        MATCH_THRESHOLD for the same vault entry — pyannote has split
        one speaker across two local IDs at a noisy boundary.

        Before pooling, an anti-merge guard checks whether any pair
        of conflicting labels was observed co-active in the same chunk.
        Co-activity is mathematical proof of distinct physical identity —
        pooling co-active speakers would silently collapse two people
        into one vault entry. The guard aborts the pool for any blocked
        group; Pass 2 handles those labels independently.

        When pooling is safe, each conflicting label receives an
        independent copy of the pooled list (not a shared reference)
        to prevent mutations to one label's list from corrupting others.

Why these run between Pass 1 and Pass 2
----------------------------------------
Both functions need the complete chunk picture. build_candidates_by_label
needs all non-overlap segments across the whole chunk to accumulate a
stable pool per label. resolve_local_conflicts needs that pool to score
mean embeddings against the vault and detect collisions. The exclusion
set also requires all segments to be present so co-activity is fully
observable.

If either ran inside the Pass 1 loop, they would evaluate partial
evidence and produce unreliable Gate 3 decisions.

No vault writes occur in this module. All operations are read-only
with respect to vault state.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..vault import SpeakerVault
from ..vault.matching import find_best_match
from ..vault.types import MATCH_THRESHOLD
from .types import _RawSegment

logger = logging.getLogger(__name__)


def build_candidates_by_label(
    raw_segments: list[_RawSegment],
) -> dict[str, list[np.ndarray]]:
    """
    Collect clean embeddings per local speaker label for Gate 3.

    Only non-overlap segments contribute. Overlap-contaminated
    embeddings would corrupt the pairwise spread check in Gate 3
    (gate_3_new_anchor_stability) and must be excluded here, not
    filtered inside the gate.

    Parameters
    ----------
    raw_segments : list[_RawSegment]
        All segments from Pass 1 for this chunk.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Local speaker label → list of clean embeddings.
        Labels with zero clean segments are absent from the dict.
    """
    candidates: dict[str, list[np.ndarray]] = defaultdict(list)
    for raw in raw_segments:
        if not raw.segment.overlap:
            candidates[raw.segment.local_speaker_label].append(raw.embedding)
    return dict(candidates)


def resolve_local_conflicts(
    raw_segments: list[_RawSegment],
    candidates_by_label: dict[str, list[np.ndarray]],
    vault: SpeakerVault,
) -> dict[str, list[np.ndarray]]:
    """
    Detect and pool candidate embeddings for conflicting local labels.

    A conflict: two chunk-local labels (e.g. SPEAKER_00, SPEAKER_01)
    both score >= MATCH_THRESHOLD against the same vault entry. This
    means pyannote split one speaker across two local IDs — a known
    failure mode at noisy chunk boundaries or during vocal variation.

    Resolution: pool all conflicting labels' candidate_embeddings under
    each conflicting label. Gate 3 now evaluates their combined evidence
    rather than each label's insufficient subset.

    Anti-merge guard
    ----------------
    Before any pooling occurs, an exclusion set is built from all
    co-active speaker pairs observed in raw_segments. If two labels
    ever appear concurrently in the same chunk, they are provably
    distinct physical speakers — co-activity is mathematical proof of
    separate identity. Pooling their embeddings would silently collapse
    two people into one vault identity.

    If any pair within a conflict group is in the exclusion set, the
    entire pool operation for that vault ID is aborted. Pass 2 handles
    the labels independently: one will win the vault match, the other
    will be flagged AMBIGUOUS or LOW_CONFIDENCE — correct forensic
    behaviour for acoustically similar but distinct speakers.

    Python reference safety
    -----------------------
    Each conflicting label receives an independent copy of the pooled
    list (`list(pooled)`), not a reference to the same object. Shared
    references would cause mutations to one label's candidate list to
    silently affect all others — a latent corruption bug in vault
    matching that would be very difficult to diagnose.

    Vault state is not modified here. All operations are read-only with
    respect to vault state.

    Parameters
    ----------
    raw_segments : list[_RawSegment]
        All segments from Pass 1.
    candidates_by_label : dict[str, list[np.ndarray]]
        Output of build_candidates_by_label().
    vault : SpeakerVault
        Live vault — read-only here.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Updated candidates_by_label. Conflicting labels that passed
        the anti-merge guard have pooled embeddings. Labels blocked
        by the guard are passed through unchanged.
    """
    if not vault.anchors:
        return candidates_by_label

    # ── Step 1: Build the anti-merge exclusion set ───────────────────────
    # Any pair of labels observed co-active in this chunk is permanently
    # banned from being pooled. Co-activity is proof of distinct identity.
    exclusion_set: set[frozenset[str]] = set()
    for raw in raw_segments:
        if raw.overlap_local_labels:
            spk_a = raw.segment.local_speaker_label
            for spk_b in raw.overlap_local_labels:
                exclusion_set.add(frozenset({spk_a, spk_b}))

    # ── Step 2: Score each label's mean embedding against vault anchors ──
    label_to_vault_match: dict[str, tuple[str, float]] = {}
    for label, embs in candidates_by_label.items():
        if not embs:
            continue
        mean_emb = np.mean(np.stack(embs), axis=0)
        best_id, best_score, _ = find_best_match(mean_emb, vault.anchors)
        if best_id is not None and best_score >= MATCH_THRESHOLD:
            label_to_vault_match[label] = (best_id, best_score)

    # ── Step 3: Group labels by matched vault ID to find collisions ──────
    vault_id_to_labels: dict[str, list[str]] = defaultdict(list)
    for label, (vault_id, _) in label_to_vault_match.items():
        vault_id_to_labels[vault_id].append(label)

    updated = dict(candidates_by_label)

    for vault_id, conflicting in vault_id_to_labels.items():
        if len(conflicting) <= 1:
            continue

        # ── Step 4: Anti-merge guard ──────────────────────────────────────
        # Check every pair in the conflict group against the exclusion set.
        blocked = False
        for i in range(len(conflicting)):
            for j in range(i + 1, len(conflicting)):
                pair = frozenset({conflicting[i], conflicting[j]})
                if pair in exclusion_set:
                    logger.critical(
                        "ANTI-MERGE GUARD: labels %s and %s both matched "
                        "vault %s but are co-active in this chunk — "
                        "they are distinct speakers. Aborting pool for "
                        "this vault ID. Pass 2 will handle independently.",
                        conflicting[i], conflicting[j], vault_id,
                    )
                    blocked = True
                    break
            if blocked:
                break

        if blocked:
            continue

        # ── Step 5: Pool (guard cleared) ─────────────────────────────────
        conflicting_sorted = sorted(
            conflicting,
            key=lambda lbl: label_to_vault_match[lbl][1],
            reverse=True,
        )
        winner = conflicting_sorted[0]
        logger.warning(
            "Local conflict on vault %s: labels %s — canonical match: %s "
            "(score=%.4f). Pooling Gate 3 candidates.",
            vault_id, conflicting, winner, label_to_vault_match[winner][1],
        )

        pooled: list[np.ndarray] = []
        for lbl in conflicting:
            pooled.extend(updated.get(lbl, []))

        # Assign independent copies — shared references would cause
        # mutations to one label's list to silently corrupt all others.
        for lbl in conflicting:
            updated[lbl] = list(pooled)

    return updated