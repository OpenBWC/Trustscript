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
        one speaker into two local IDs at a noisy boundary.
        Resolution: pool the conflicting labels' candidate embeddings.
        Both will naturally match the same vault ID via cosine scoring.
        Pooling prevents false INSUFFICIENT_SEGMENTS rejections.

Why these run between Pass 1 and Pass 2
----------------------------------------
Both functions need the complete chunk picture. build_candidates_by_label
needs all non-overlap segments across the whole chunk to accumulate a
stable pool per label. resolve_local_conflicts needs that pool to score
mean embeddings against the vault and detect collisions.

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

    Vault state is not modified. This is pre-matching preparation only.

    If the vault is empty (first chunk before any seeding), returns
    candidates_by_label unchanged — no conflicts are possible.

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
        Updated candidates_by_label with pooled embeddings for any
        conflicting labels. Unchanged labels are passed through as-is.
    """
    if not vault.anchors:
        return candidates_by_label

    # Score each label's mean embedding against all vault anchors.
    label_to_vault_match: dict[str, tuple[str, float]] = {}
    for label, embs in candidates_by_label.items():
        if not embs:
            continue
        mean_emb = np.mean(np.stack(embs), axis=0)
        best_id, best_score, _ = find_best_match(mean_emb, vault.anchors)
        if best_id is not None and best_score >= MATCH_THRESHOLD:
            label_to_vault_match[label] = (best_id, best_score)

    # Group labels by their matched vault ID to find collisions.
    vault_id_to_labels: dict[str, list[str]] = defaultdict(list)
    for label, (vault_id, _) in label_to_vault_match.items():
        vault_id_to_labels[vault_id].append(label)

    updated = dict(candidates_by_label)

    for vault_id, conflicting in vault_id_to_labels.items():
        if len(conflicting) <= 1:
            continue

        # Sort descending by score — the label closest to the vault anchor
        # is the stronger match and treated as the canonical winner.
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

        # Pool all conflicting labels' candidates under every conflicting label.
        # Each label now has the full evidence set for Gate 3 evaluation.
        pooled: list[np.ndarray] = []
        for lbl in conflicting:
            pooled.extend(updated.get(lbl, []))
        for lbl in conflicting:
            updated[lbl] = pooled

    return updated