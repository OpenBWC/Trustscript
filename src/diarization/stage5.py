"""
src/diarization/stage5.py
==========================
TrustScript Phase 1 — Stage 5 Orchestrator: Windowed Diarization + Vault Matching

Responsibility
--------------
Orchestrate Stage 5 by:
    1. Deciding vault initialization strategy based on file-level SNR.
    2. Instantiating the SpeakerVault.
    3. Calling run_windowed_diarization() with the loaded models.
    4. Applying post-processing flags (LOW_ANCHOR_CONFIDENCE, quality windows).
    5. Computing the refined SNR estimate from pyannote's VAD output.
    6. Writing the intermediate timeline JSON to disk.
    7. Assembling and returning Stage5Result.

Stage 5 is the most computationally intensive stage in Phase 1.
All ML inference runs here via the engine. This module contains no ML
code — it owns only orchestration, flag application, and I/O.

Vault initialization strategy
-------------------------------
The vault always starts empty. No manual pre-seeding occurs in this
module. The engine's match_or_create() handles all vault writes through
the gate system.

The initialization decision made here is whether to apply blanket
LOW_ANCHOR_CONFIDENCE flagging:

    File-level SNR POOR or CRITICAL (< 10.0 dB):
        low_anchor_confidence = True
        After the engine completes, every first-chunk segment receives
        LOW_ANCHOR_CONFIDENCE + PROVISIONAL. Stage 6 re-scores these
        against the mature vault.

    File-level SNR FAIR or better (>= 10.0 dB):
        low_anchor_confidence = False
        The engine's vault gate logic handles per-segment quality
        decisions (Gates 1–4). No blanket flagging.

    File-level SNR unavailable (None):
        Treated conservatively as chaotic — low_anchor_confidence = True.

Quality window flagging
------------------------
After the engine runs, segments whose midpoints fall within flagged
quality windows (LOW_SNR, UNSTABLE_NOISE, CLIPPING from Stage 2) receive
PROVISIONAL if not already set. Midpoint-based matching is consistent
with engine.py's seam management — one ownership rule throughout.

Intermediate timeline file
---------------------------
Stage 5 writes {incident_id}_timeline.json with:
    status: "stage5_complete"

Stage 6 reads this file, re-scores PROVISIONAL segments against the
mature vault, and writes it back with status: "stage6_complete".
Stage 7 reads the finalized file for output assembly.

The status field makes any crashed run recoverable — you can always
determine exactly which stage last touched the file.

Refined SNR
-----------
After all chunks are processed, a refined SNR estimate is computed
using pyannote's VAD output (speech_intervals from EngineResult) as
window labels on the normalized audio. More accurate than Stage 2's
energy-based estimate for challenging BWC conditions.

Both values are preserved in output:
    audio_quality.snr_db            Stage 2 pre-normalization estimate
    quality_summary.snr_db_refined  Stage 5 post-normalization estimate

Downstream phases should prefer the refined value for Fusion weighting.
The Stage 2 value remains useful for understanding original recording
quality before normalization was applied.

Contributors
------------
    Do not add ML inference code here — that belongs in engine/.
    Do not add vault gate logic here — that belongs in gates.py.
    Do not add centroid update logic here — that belongs in vault/.
    This module owns only orchestration, flag post-processing, and I/O.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from .engine import DEFAULT_CHUNK_SIZE, EngineResult, run_windowed_diarization
from .models import Stage4Result
from .segment import (
    FLAG_LOW_ANCHOR_CONFIDENCE,
    FLAG_PROVISIONAL,
    TimelineSegment,
)
from .vault import SpeakerVault

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: File-level SNR below this triggers LOW_ANCHOR_CONFIDENCE on all first-chunk
#: segments. Corresponds to the POOR/FAIR classification boundary.
_CHAOTIC_SNR_THRESHOLD: float = 10.0

#: SNR classification thresholds. Ordered highest-first — first match wins.
_SNR_CLASSIFICATIONS: list[tuple[float, str]] = [
    (30.0, "EXCELLENT"),
    (20.0, "GOOD"),
    (10.0, "FAIR"),
    (0.0,  "POOR"),
    (float("-inf"), "CRITICAL"),
]

#: Prevents log10(0) in SNR computation.
_SNR_EPSILON: float = 1e-10

#: Intermediate and vault output file suffixes.
_TIMELINE_SUFFIX: str = "_timeline.json"
_VAULT_SUFFIX: str = "_vault.json"

#: Format version for intermediate timeline files.
#: Increment when the schema changes to allow Stage 6 to detect stale files.
_SCHEMA_VERSION: str = "1.0"

#: Status tag written by this stage. Stage 6 asserts this before reading.
_STATUS_STAGE5: str = "stage5_complete"


# ---------------------------------------------------------------------------
# Stage 5 result
# ---------------------------------------------------------------------------

@dataclass
class Stage5Result:
    """
    Complete output of Stage 5: Windowed Diarization + Vault Matching.

    Passed to Stage 6 (Retroactive Re-scoring) and Stage 7 (Output Assembly).

    Attributes
    ----------
    timeline : list[TimelineSegment]
        All diarized segments in chronological order. Speaker IDs are
        global (TRUST_SPK_NN). PROVISIONAL segments are candidates for
        Stage 6 re-scoring.
    vault : SpeakerVault
        Fully populated vault after all chunks are processed.
        Passed to Stage 6 for re-scoring PROVISIONAL segments against
        mature centroids.
    vault_metadata : dict
        Serialized vault state from SpeakerVault.get_vault_metadata().
        Contains "speakers", "vault_quality", and "vault_detail" blocks.
    intermediate_timeline_path : Path
        Path to the written {incident_id}_timeline.json.
        Stage 6 reads and re-writes this file.
    chunk_count : int
        Number of chunks processed by the engine.
    total_speech_seconds : float
        Total non-overlap speech duration across the full file.
    snr_db_refined : float | None
        Refined SNR estimate computed from pyannote's VAD labels on the
        normalized audio. None if computation failed or no speech found.
    snr_db_refined_classification : str | None
        SNR classification of the refined estimate.
        One of: EXCELLENT / GOOD / FAIR / POOR / CRITICAL.
        None when snr_db_refined is None.
    low_anchor_confidence : bool
        True when file-level SNR was below _CHAOTIC_SNR_THRESHOLD.
        All first-chunk segments carry LOW_ANCHOR_CONFIDENCE in this case.
    """
    timeline: list[TimelineSegment]
    vault: SpeakerVault
    vault_metadata: dict
    intermediate_timeline_path: Path
    chunk_count: int
    total_speech_seconds: float
    snr_db_refined: float | None
    snr_db_refined_classification: str | None
    low_anchor_confidence: bool

    def to_processing_notes(self) -> dict:
        """
        Serialize Stage 5 results for the processing_notes block.
        Called by Stage 7 (Output Assembly).
        """
        provisional_count = sum(1 for seg in self.timeline if seg.is_provisional)
        return {
            "stage_5": {
                "chunk_count": self.chunk_count,
                "total_speech_seconds": round(self.total_speech_seconds, 3),
                "total_segments": len(self.timeline),
                "provisional_segments": provisional_count,
                "low_anchor_confidence": self.low_anchor_confidence,
                "snr_db_refined": (
                    round(self.snr_db_refined, 2)
                    if self.snr_db_refined is not None else None
                ),
                "snr_db_refined_classification": self.snr_db_refined_classification,
                "snr_refined_method": "pyannote_vad",
                "vault_quality": self.vault_metadata.get("vault_quality", {}),
                "intermediate_timeline": str(self.intermediate_timeline_path),
            }
        }


# ---------------------------------------------------------------------------
# Stage 5 entry point
# ---------------------------------------------------------------------------

def run_stage5(
    stage2_result: Any,
    stage3_result: Any,
    stage4_result: Stage4Result,
    output_dir: Path,
    incident_id: str,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
) -> Stage5Result:
    """
    Execute Stage 5: Windowed Diarization + Vault Matching.

    Parameters
    ----------
    stage2_result
        Complete output from run_stage2(). Provides file-level SNR for
        the vault initialization decision and QualityWindows for
        post-processing flag application.
        Expected attributes:
            .audio_quality.snr_db             float | None
            .audio_quality.snr_classification str
            .quality_windows                  list[QualityWindow]
    stage3_result
        Complete output from run_stage3(). Provides the normalized
        audio path for engine processing and refined SNR computation.
        Expected attributes:
            .normalized_path                  Path (pcm_s16le mono WAV)
    stage4_result : Stage4Result
        Loaded pyannote models from Stage 4. Passed directly to the engine.
    output_dir : Path
        Directory to write the intermediate timeline JSON.
        Created if it does not exist.
    incident_id : str
        Incident identifier used to name output files.
    chunk_size : float
        Chunk duration in seconds. Default: 300.0 (5 minutes).
        Must be > CHUNK_OVERLAP (10.0s) — engine raises ValueError if not.

    Returns
    -------
    Stage5Result
        Complete Stage 5 output. Pass to Stage 6 for PROVISIONAL
        re-scoring, then Stage 7 for final output assembly.
    """
    normalized_path: Path = stage3_result.normalized_path
    logger.info(
        "Stage 5 — Windowed Diarization + Vault Matching: %s",
        normalized_path.name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint interception.
    # If a timeline file already exists with a valid status, skip the
    # engine entirely. This prevents re-running hours of diarization
    # when the pipeline crashed in Stage 6 or later.
    #
    # Valid resume statuses:
    #   stage5_complete  — Stage 5 finished, Stage 6 not yet started.
    #   stage6_complete  — Stage 6 re-scored the timeline.
    #   stage7_complete  — Full pipeline completed.
    #
    # Vault limitation: the vault cannot currently be restored from disk
    # on resume. The returned Stage5Result carries an empty vault.
    # Stage 6 reads the timeline file directly and does not depend on
    # the in-memory vault for its re-scoring pass. Full vault resumption
    # will be implemented when _vault.json deserialization is added.
    #
    # Corruption handling: a partially written file (interrupted write)
    # will fail json.load() and trigger a warning + full re-run.
    # The atomic write pattern in _write_timeline_json prevents this
    # from occurring on writes produced by this version of the code.
    # ------------------------------------------------------------------
    timeline_path = output_dir / f"{incident_id}{_TIMELINE_SUFFIX}"
    resume_result = _try_resume_from_checkpoint(
        timeline_path=timeline_path,
        incident_id=incident_id,
    )
    if resume_result is not None:
        return resume_result

    # ------------------------------------------------------------------
    # Vault initialization decision.
    # Read file-level SNR from Stage 2 to decide whether the opening
    # is too chaotic for confident vault seeding.
    # ------------------------------------------------------------------
    snr_db: float | None = getattr(
        getattr(stage2_result, "audio_quality", None), "snr_db", None
    )
    low_anchor_confidence = _is_chaotic_opening(snr_db)

    if low_anchor_confidence:
        logger.warning(
            "Stage 5: file-level SNR %.1f dB is below chaotic threshold "
            "(%.1f dB). All first-chunk segments will be flagged "
            "LOW_ANCHOR_CONFIDENCE + PROVISIONAL after engine completes.",
            snr_db if snr_db is not None else float("nan"),
            _CHAOTIC_SNR_THRESHOLD,
        )

    # Vault starts empty. Engine's match_or_create() handles all seeding
    # through the gate system. No manual pre-seeding in this module.
    vault = SpeakerVault()

    # ------------------------------------------------------------------
    # Engine execution.
    # All ML inference runs here. torch.no_grad() is enforced inside
    # run_windowed_diarization() — see engine/engine.py.
    # ------------------------------------------------------------------
    logger.info(
        "Stage 5: starting windowed diarization (chunk_size=%.0fs).",
        chunk_size,
    )
    engine_result: EngineResult = run_windowed_diarization(
        audio_path=normalized_path,
        stage4_result=stage4_result,
        vault=vault,
        chunk_size=chunk_size,
    )
    timeline = engine_result.timeline
    logger.info(
        "Stage 5: engine complete — %d segments, %d chunks, %.1fs speech.",
        len(timeline), engine_result.chunk_count, engine_result.total_speech_seconds,
    )

    # ------------------------------------------------------------------
    # Post-processing pass 1: LOW_ANCHOR_CONFIDENCE blanket flagging.
    # Applied to all chunk_index=0 segments when the opening was chaotic.
    # ------------------------------------------------------------------
    if low_anchor_confidence:
        _apply_low_anchor_confidence_flags(timeline)

    # ------------------------------------------------------------------
    # Post-processing pass 2: quality window PROVISIONAL flagging.
    # Segments in Stage 2 flagged quality windows are marked PROVISIONAL
    # so Stage 6 can re-evaluate them against the mature vault.
    # ------------------------------------------------------------------
    quality_windows = getattr(stage2_result, "quality_windows", [])
    if quality_windows:
        _apply_quality_window_flags(timeline, quality_windows)
    else:
        logger.debug(
            "Stage 5: no quality_windows on stage2_result — "
            "skipping quality window provisional flagging."
        )

    # ------------------------------------------------------------------
    # Refined SNR computation.
    # Uses pyannote's VAD output (speech_intervals) as window labels.
    # More accurate than Stage 2's energy-based estimate for BWC audio.
    # ------------------------------------------------------------------
    snr_db_refined, snr_db_refined_classification = _compute_refined_snr(
        normalized_path=normalized_path,
        speech_intervals=engine_result.speech_intervals,
    )
    if snr_db_refined is not None:
        logger.info(
            "Stage 5: refined SNR = %.1f dB (%s) via pyannote VAD.",
            snr_db_refined, snr_db_refined_classification,
        )
    else:
        logger.warning(
            "Stage 5: refined SNR could not be computed — "
            "insufficient speech or audio I/O failure. "
            "Downstream phases will use Stage 2 SNR estimate."
        )

    # ------------------------------------------------------------------
    # Vault metadata assembly.
    # ------------------------------------------------------------------
    vault_metadata = vault.get_vault_metadata()
    logger.info(
        "Stage 5: vault — %d speakers, purity=%.3f.",
        len(vault_metadata.get("speakers", [])),
        vault_metadata.get("vault_quality", {}).get("vault_purity_estimate", 0.0),
    )

    # ------------------------------------------------------------------
    # Write intermediate timeline JSON.
    # Stage 6 reads this file. Stage 7 reads Stage 6's updated version.
    # ------------------------------------------------------------------
    timeline_path = output_dir / f"{incident_id}{_TIMELINE_SUFFIX}"
    _write_timeline_json(
        timeline=timeline,
        vault_metadata=vault_metadata,
        output_path=timeline_path,
        incident_id=incident_id,
        chunk_count=engine_result.chunk_count,
        total_speech_seconds=engine_result.total_speech_seconds,
        low_anchor_confidence=low_anchor_confidence,
        snr_db_refined=snr_db_refined,
        snr_db_refined_classification=snr_db_refined_classification,
    )
    logger.info(
        "Stage 5: timeline written → %s (%.1f KB).",
        timeline_path.name,
        timeline_path.stat().st_size / 1024,
    )

    provisional_count = sum(1 for seg in timeline if seg.is_provisional)
    logger.info(
        "Stage 5 complete — %d segments | %d speakers | "
        "%d PROVISIONAL | low_anchor_confidence=%s.",
        len(timeline),
        len(vault_metadata.get("speakers", [])),
        provisional_count,
        low_anchor_confidence,
    )

    return Stage5Result(
        timeline=timeline,
        vault=vault,
        vault_metadata=vault_metadata,
        intermediate_timeline_path=timeline_path,
        chunk_count=engine_result.chunk_count,
        total_speech_seconds=engine_result.total_speech_seconds,
        snr_db_refined=snr_db_refined,
        snr_db_refined_classification=snr_db_refined_classification,
        low_anchor_confidence=low_anchor_confidence,
    )


# ---------------------------------------------------------------------------
# Checkpoint resumption
# ---------------------------------------------------------------------------

#: Timeline file statuses that indicate Stage 5 has already completed.
_RESUMABLE_STATUSES: frozenset[str] = frozenset({
    "stage5_complete",
    "stage6_complete",
    "stage7_complete",
})


def _try_resume_from_checkpoint(
    timeline_path: Path,
    incident_id: str,
) -> Stage5Result | None:
    """
    Attempt to resume Stage 5 from an existing timeline file.

    Returns a Stage5Result reconstructed from disk if a valid checkpoint
    exists. Returns None if no checkpoint is found, the file is missing,
    or the file is corrupted — in all None cases the caller proceeds with
    a full engine run.

    Vault limitation
    ----------------
    The returned Stage5Result carries an empty SpeakerVault. The vault
    cannot currently be restored from disk because _vault.json
    deserialization is not yet implemented. Stage 6 reads the timeline
    file directly and does not require the in-memory vault for re-scoring,
    so this is acceptable for resume purposes.

    Parameters
    ----------
    timeline_path : Path
        Expected location of {incident_id}_timeline.json.
    incident_id : str
        Used for log messages only.

    Returns
    -------
    Stage5Result | None
        Reconstructed result if a valid checkpoint exists, else None.
    """
    if not timeline_path.exists():
        return None

    try:
        data = json.loads(timeline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Stage 5: corrupted or unreadable checkpoint at %s (%s). "
            "Overwriting with a fresh run.",
            timeline_path.name, exc,
        )
        return None

    status = data.get("status", "")
    if status not in _RESUMABLE_STATUSES:
        logger.debug(
            "Stage 5: checkpoint status %r is not resumable — "
            "proceeding with full run.",
            status,
        )
        return None

    logger.info(
        "Stage 5: valid checkpoint found for %s (status=%r). "
        "Skipping diarization engine.",
        incident_id, status,
    )

    # Deserialize timeline segments from the checkpoint.
    raw_segments = data.get("timeline", [])
    timeline = [TimelineSegment.from_dict(s) for s in raw_segments]

    # Reconstruct vault_metadata from what was written.
    vault_metadata = {
        "speakers": data.get("speakers", []),
        "vault_quality": data.get("vault_quality", {}),
        "vault_detail": {},  # Not persisted in timeline.json.
    }

    quality_summary = data.get("quality_summary", {})

    return Stage5Result(
        timeline=timeline,
        vault=SpeakerVault(),       # Empty — vault not yet serialized to disk.
        vault_metadata=vault_metadata,
        intermediate_timeline_path=timeline_path,
        chunk_count=data.get("chunk_count", 0),
        total_speech_seconds=data.get("total_speech_seconds", 0.0),
        snr_db_refined=quality_summary.get("snr_db_refined"),
        snr_db_refined_classification=quality_summary.get("snr_db_refined_classification"),
        low_anchor_confidence=data.get("low_anchor_confidence", False),
    )


# ---------------------------------------------------------------------------
# Vault initialization helpers
# ---------------------------------------------------------------------------

def _is_chaotic_opening(snr_db: float | None) -> bool:
    """
    Return True if the file-level SNR indicates a chaotic opening.

    A None SNR (could not be measured by Stage 2) is treated
    conservatively as chaotic — absence of a quality measurement
    is not evidence of good quality.

    Parameters
    ----------
    snr_db : float | None
        File-level SNR from Stage 2. None if unavailable.

    Returns
    -------
    bool
        True when snr_db is None or below _CHAOTIC_SNR_THRESHOLD (10.0 dB).
    """
    if snr_db is None:
        return True
    return snr_db < _CHAOTIC_SNR_THRESHOLD


def _apply_low_anchor_confidence_flags(timeline: list[TimelineSegment]) -> None:
    """
    Add LOW_ANCHOR_CONFIDENCE and PROVISIONAL to all first-chunk segments.

    Called when the file-level SNR indicates a chaotic opening. Both flags
    are required: LOW_ANCHOR_CONFIDENCE signals the cause, PROVISIONAL
    marks the segment as a Stage 6 re-scoring candidate.

    Only segments with chunk_index == 0 are affected. Idempotent — flags
    are only appended if not already present.

    Parameters
    ----------
    timeline : list[TimelineSegment]
        Full diarization timeline from the engine. Mutated in place.
    """
    flagged = 0
    for seg in timeline:
        if seg.chunk_index == 0:
            if FLAG_LOW_ANCHOR_CONFIDENCE not in seg.flags:
                seg.flags.append(FLAG_LOW_ANCHOR_CONFIDENCE)
            if FLAG_PROVISIONAL not in seg.flags:
                seg.flags.append(FLAG_PROVISIONAL)
            flagged += 1
    logger.debug(
        "_apply_low_anchor_confidence_flags: flagged %d first-chunk segments.",
        flagged,
    )


# ---------------------------------------------------------------------------
# Quality window flagging
# ---------------------------------------------------------------------------

def _apply_quality_window_flags(
    timeline: list[TimelineSegment],
    quality_windows: list,
) -> None:
    """
    Add PROVISIONAL to segments whose midpoints fall in degraded windows.

    A segment is considered to fall within a flagged window if:
        1. Its midpoint (mean of start_seconds and end_seconds) falls
           within the window's [start_seconds, end_seconds) range.
        2. The window carries at least one quality flag
           (LOW_SNR, UNSTABLE_NOISE, or CLIPPING).

    Midpoint-based matching is consistent with engine.py's seam
    management — one ownership rule throughout the pipeline.

    Does not overwrite existing flags. Appends PROVISIONAL only if
    absent. Breaks after the first matching window per segment —
    one match is sufficient.

    Parameters
    ----------
    timeline : list[TimelineSegment]
        Full diarization timeline. Mutated in place.
    quality_windows : list[QualityWindow]
        30-second quality windows from Stage 2. Each window must have:
            .start_seconds : float
            .end_seconds   : float
            .flags         : list[str]
    """
    # Build flagged ranges once — avoids repeated attribute access per segment.
    flagged_ranges: list[tuple[float, float]] = [
        (w.start_seconds, w.end_seconds)
        for w in quality_windows
        if getattr(w, "flags", [])
    ]

    if not flagged_ranges:
        logger.debug(
            "_apply_quality_window_flags: no flagged windows in stage2_result — "
            "no PROVISIONAL flags added."
        )
        return

    added = 0
    for seg in timeline:
        midpoint = (seg.start_seconds + seg.end_seconds) / 2.0
        for win_start, win_end in flagged_ranges:
            if win_start <= midpoint < win_end:
                if FLAG_PROVISIONAL not in seg.flags:
                    seg.flags.append(FLAG_PROVISIONAL)
                    added += 1
                break  # One match is sufficient — skip remaining windows.

    logger.debug(
        "_apply_quality_window_flags: added PROVISIONAL to %d segments "
        "in degraded quality windows.",
        added,
    )


# ---------------------------------------------------------------------------
# Refined SNR
# ---------------------------------------------------------------------------

def _compute_refined_snr(
    normalized_path: Path,
    speech_intervals: list[tuple[float, float]],
) -> tuple[float | None, str | None]:
    """
    Compute a refined SNR estimate using pyannote's VAD output as labels.

    Algorithm:
        1. Load the normalized audio (pcm_s16le WAV — sequential read,
           no chunking required since we need aggregate RMS only).
        2. Build a boolean speech mask using speech_intervals.
        3. Compute RMS for speech frames and non-speech (noise) frames.
        4. SNR (dB) = 20 * log10(speech_rms / noise_rms).

    This is more accurate than Stage 2's energy-based estimate for
    challenging BWC conditions where energy-based VAD struggles
    (overlapping speech, radio chatter, whispered speech).

    The read is performed under torch.no_grad() for consistency with
    the engine's memory contract, though no gradients are computed here.

    Parameters
    ----------
    normalized_path : Path
        Stage 3 normalized pcm_s16le mono WAV.
    speech_intervals : list[tuple[float, float]]
        (start_seconds, end_seconds) of non-overlap speech from EngineResult.
        Empty list returns (None, None) immediately.

    Returns
    -------
    tuple[float | None, str | None]
        (snr_db_refined, snr_classification)
        Both None if computation failed or data was insufficient.
    """
    if not speech_intervals:
        logger.warning(
            "_compute_refined_snr: no speech intervals — "
            "cannot compute refined SNR."
        )
        return None, None

    try:
        with torch.no_grad():
            waveform, sample_rate = torchaudio.load(str(normalized_path))

        # Guarantee mono — Stage 3 should have ensured this already.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform_np: np.ndarray = waveform.squeeze(0).numpy()
        num_samples = len(waveform_np)

        # Build boolean speech mask from speech_intervals.
        speech_mask = np.zeros(num_samples, dtype=bool)
        for start_sec, end_sec in speech_intervals:
            start_sample = int(start_sec * sample_rate)
            end_sample = min(int(end_sec * sample_rate), num_samples)
            if end_sample > start_sample:
                speech_mask[start_sample:end_sample] = True

        speech_samples = waveform_np[speech_mask]
        noise_samples = waveform_np[~speech_mask]

        if len(speech_samples) == 0 or len(noise_samples) == 0:
            logger.warning(
                "_compute_refined_snr: speech=%d samples, noise=%d samples — "
                "one region is empty, cannot compute SNR.",
                len(speech_samples), len(noise_samples),
            )
            return None, None

        speech_rms = float(np.sqrt(np.mean(speech_samples ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise_samples ** 2)))

        if noise_rms < _SNR_EPSILON:
            logger.warning(
                "_compute_refined_snr: noise floor is effectively silent "
                "(rms=%.2e) — SNR estimate would be unreliable.",
                noise_rms,
            )
            return None, None

        snr_db = float(20.0 * np.log10((speech_rms / noise_rms) + _SNR_EPSILON))
        return round(snr_db, 2), _classify_snr(snr_db)

    except Exception:
        logger.exception(
            "_compute_refined_snr: failed to compute refined SNR from %s.",
            normalized_path.name,
        )
        return None, None


def _classify_snr(snr_db: float) -> str:
    """
    Return the SNR classification string for a given value in dB.

    Thresholds (highest-first, first match wins):
        >= 30 dB  → EXCELLENT
        >= 20 dB  → GOOD
        >= 10 dB  → FAIR
        >= 0 dB   → POOR
        <  0 dB   → CRITICAL
    """
    for threshold, label in _SNR_CLASSIFICATIONS:
        if snr_db >= threshold:
            return label
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Intermediate timeline JSON
# ---------------------------------------------------------------------------

def _write_timeline_json(
    timeline: list[TimelineSegment],
    vault_metadata: dict,
    output_path: Path,
    incident_id: str,
    chunk_count: int,
    total_speech_seconds: float,
    low_anchor_confidence: bool,
    snr_db_refined: float | None,
    snr_db_refined_classification: str | None,
) -> None:
    """
    Write the intermediate timeline JSON to disk.

    Stage 6 reads this file to re-score PROVISIONAL segments.
    Stage 7 reads the Stage-6-updated version for final output assembly.

    The status field ("stage5_complete") allows any stage to verify the
    file was written by the correct producer. A crashed pipeline is
    recoverable — the status tells you exactly where to resume.

    File structure
    --------------
    {
        "schema_version": "1.0",
        "status": "stage5_complete",
        "incident_id": "...",
        "generated_at": "<ISO 8601 UTC>",
        "chunk_count": 5,
        "total_speech_seconds": 234.5,
        "low_anchor_confidence": false,
        "quality_summary": {
            "snr_db_refined": 13.1,
            "snr_db_refined_classification": "FAIR",
            "snr_refined_method": "pyannote_vad"
        },
        "vault_quality": { ... },
        "speakers": [ ... ],
        "timeline": [ { segment ... }, ... ]
    }

    Parameters
    ----------
    timeline : list[TimelineSegment]
        All segments, serialized via TimelineSegment.to_dict().
    vault_metadata : dict
        Output of SpeakerVault.get_vault_metadata().
    output_path : Path
        Destination file. Parent directory must already exist.
    incident_id : str
        Written to the header for traceability.
    chunk_count : int
        Number of chunks processed. Written to header.
    total_speech_seconds : float
        Total non-overlap speech duration. Written to header.
    low_anchor_confidence : bool
        Written to header. Stage 6 reads this to decide re-scoring scope.
    snr_db_refined, snr_db_refined_classification : float|None, str|None
        Written to quality_summary block.
    """
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": _STATUS_STAGE5,
        "incident_id": incident_id,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "chunk_count": chunk_count,
        "total_speech_seconds": round(total_speech_seconds, 3),
        "low_anchor_confidence": low_anchor_confidence,
        "quality_summary": {
            "snr_db_refined": snr_db_refined,
            "snr_db_refined_classification": snr_db_refined_classification,
            "snr_refined_method": "pyannote_vad",
        },
        "vault_quality": vault_metadata.get("vault_quality", {}),
        "speakers": vault_metadata.get("speakers", []),
        "timeline": [seg.to_dict() for seg in timeline],
    }

    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        # Atomic rename — the target file is never observed in a partial
        # state. On POSIX this is a single syscall (rename(2)). On Windows
        # it is atomic on the same volume. A crash before this line leaves
        # the .tmp file on disk, which the checkpoint check ignores.
        tmp_path.replace(output_path)
    except Exception:
        # Clean up the temp file if the write or rename failed.
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise