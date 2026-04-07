"""
src/utils/normalization/stage3.py
===================================
TrustScript Phase 1 — Stage 3 Orchestrator: Normalization

Responsibility
--------------
Coordinate Stage 3 by calling filter.py's two-pass normalization and
assembling the Stage3Result. Contains the skip-if-passthrough logic
and the known-bias documentation.

Passthrough behaviour
---------------------
If Stage 1 determined the file already meets all normalization targets
(16kHz, mono, PCM, -16 LUFS ± 1), Stage 3 is skipped entirely. The
working audio path from the extraction step is returned as the
normalized path — no new file is written and no ffmpeg process runs.

This is recorded in the output with:
    normalization_skipped: true
    normalization_applied: []

Known biases
------------
Every transformation applied to the signal is documented in the output
as a known_biases list. The applicable biases depend on the source file:

    Always:         HIGH_FREQ_LOSS, ANTIALIAS_ATTENUATION,
                    LOUDNESS_NORMALIZATION
    Stereo sources: MONO_DOWNMIX (additional)
    Passthrough:    [] (no transformations applied)

These are written to processing_notes.stage_3.known_biases in the
final output JSON so downstream consumers know what was done to the
signal before pyannote processed it.

Contributors
------------
    Do not add ffmpeg subprocess logic here — that belongs in filter.py.
    Do not add bias label strings here — those belong in constants.py.
    This module owns only orchestration and result assembly.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    BIAS_ANTIALIAS_ATTENUATION,
    BIAS_HIGH_FREQ_LOSS,
    BIAS_LOUDNESS_NORMALIZATION,
    BIAS_MONO_DOWNMIX,
    NORMALIZED_SUFFIX,
    TARGET_CHANNELS,
    TARGET_LUFS,
    TARGET_SAMPLE_RATE,
)
from .filter import LoudnormMeasurement, apply_normalization, measure_loudness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 3 result
# ---------------------------------------------------------------------------


@dataclass
class Stage3Result:
    """
    Complete output of Stage 3: Normalization.

    Passed to Stage 4 (Model Loading) and Stage 7 (Output Assembly).

    Attributes
    ----------
    normalized_path : Path
        Path to the normalized 16kHz mono WAV ready for pyannote.
        When normalization was skipped (passthrough), this is the
        working audio path from extraction — no new file was written.
    normalization_skipped : bool
        True when Stage 1 passthrough conditions were met and no
        processing was applied.
    normalization_applied : list[str]
        Operations applied during normalization in execution order.
        Empty when normalization_skipped is True.
        Possible values: "mono_downmix", "resample_16khz", "loudnorm_r128"
    known_biases : list[str]
        Human-readable descriptions of signal transformations applied.
        Written verbatim to processing_notes.stage_3.known_biases.
        Empty when normalization_skipped is True.
    loudnorm_measurement : LoudnormMeasurement | None
        Pass 1 loudness measurements from the analysis pass.
        None when normalization_skipped is True.
    output_sample_rate : int
        Sample rate of the normalized output (always 16000).
    output_channels : int
        Channel count of the normalized output (always 1).
    """
    normalized_path: Path
    normalization_skipped: bool
    normalization_applied: list[str] = field(default_factory=list)
    known_biases: list[str] = field(default_factory=list)
    loudnorm_measurement: LoudnormMeasurement | None = None
    output_sample_rate: int = TARGET_SAMPLE_RATE
    output_channels: int = TARGET_CHANNELS

    def to_processing_notes(self) -> dict:
        """
        Serialize Stage 3 results into the processing_notes dict format
        for the TrustScript output JSON.

        Called by Stage 7 (Output Assembly, src/utils/schema.py).
        """
        notes: dict = {
            "normalization_skipped": self.normalization_skipped,
            "normalization_applied": self.normalization_applied,
            "known_biases": self.known_biases,
            "output_sample_rate": self.output_sample_rate,
            "output_channels": self.output_channels,
        }

        if self.loudnorm_measurement is not None:
            notes["loudnorm_pass1"] = {
                "input_i_lufs": self.loudnorm_measurement.input_i,
                "input_lra_lu": self.loudnorm_measurement.input_lra,
                "input_tp_dbtp": self.loudnorm_measurement.input_tp,
                "target_offset_db": self.loudnorm_measurement.target_offset,
                "target_i_lufs": TARGET_LUFS,
            }

        return {"stage_3": notes}


# ---------------------------------------------------------------------------
# Stage 3 entry point
# ---------------------------------------------------------------------------


def run_stage3(
    stage1_result: "Stage1Result",  # type: ignore[name-defined]
    working_audio_path: Path,
    output_dir: Path,
    incident_id: str,
) -> Stage3Result:
    """
    Execute Stage 3: Normalization.

    Skips entirely when Stage 1 determined the file already meets all
    normalization targets (passthrough mode). Otherwise runs a two-pass
    ffmpeg loudnorm to resample, downmix, and normalise the audio.

    The normalized output is written to:
        {output_dir}/{incident_id}_normalized.wav

    Parameters
    ----------
    stage1_result : Stage1Result
        Complete output from run_stage1(). Provides the passthrough
        decision and original audio properties.
    working_audio_path : Path
        Working audio file from extract_working_audio(). This is the
        input to the normalization filter chain.
    output_dir : Path
        Directory to write the normalized WAV. Created if it does not exist.
    incident_id : str
        Incident identifier used to name the output file.

    Returns
    -------
    Stage3Result
        Complete Stage 3 results including the normalized file path.
        Pass normalized_path to Stage 4 (Model Loading) and all
        subsequent stages that process audio.
    """
    from ..audio.stage1 import Stage1Result  # Avoid circular import.

    props = stage1_result.audio_properties
    passthrough = stage1_result.passthrough

    logger.info("Stage 3 — Normalization: %s", stage1_result.original_filename)

    # ------------------------------------------------------------------
    # Passthrough: file already meets all targets — skip normalization.
    # ------------------------------------------------------------------
    if passthrough.can_skip_normalization:
        logger.info(
            "Stage 3 skipped — file already meets all normalization targets. "
            "Using working audio path as normalized path."
        )
        return Stage3Result(
            normalized_path=working_audio_path,
            normalization_skipped=True,
        )

    # ------------------------------------------------------------------
    # Prepare output path.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / f"{incident_id}{NORMALIZED_SUFFIX}"

    # ------------------------------------------------------------------
    # Pass 1: Measure loudness of the working audio.
    # ------------------------------------------------------------------
    logger.info("Pass 1 — measuring loudness: %s", working_audio_path.name)
    try:
        measurement = measure_loudness(working_audio_path)
    except RuntimeError as e:
        raise RuntimeError(
            f"Stage 3 Pass 1 failed for {stage1_result.original_filename}: {e}"
        ) from e

    # ------------------------------------------------------------------
    # Pass 2: Apply normalization filter chain.
    # ------------------------------------------------------------------
    logger.info(
        "Pass 2 — applying normalization: %s → %s",
        working_audio_path.name, normalized_path.name,
    )
    try:
        apply_normalization(
            source=working_audio_path,
            dest=normalized_path,
            measurement=measurement,
            source_channels=props.channels,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Stage 3 Pass 2 failed for {stage1_result.original_filename}: {e}"
        ) from e

    # ------------------------------------------------------------------
    # Document what was applied and the known biases introduced.
    # ------------------------------------------------------------------
    normalization_applied: list[str] = []
    known_biases: list[str] = []

    if props.channels > 1:
        normalization_applied.append("mono_downmix")
        known_biases.append(BIAS_MONO_DOWNMIX)

    normalization_applied.append("resample_16khz")
    known_biases.extend([
        BIAS_HIGH_FREQ_LOSS,
        BIAS_ANTIALIAS_ATTENUATION,
    ])

    normalization_applied.append("loudnorm_r128")
    known_biases.append(BIAS_LOUDNESS_NORMALIZATION)

    logger.info(
        "Stage 3 complete — %s (%.1f MB) | applied: %s",
        normalized_path.name,
        normalized_path.stat().st_size / 1_048_576,
        ", ".join(normalization_applied),
    )

    return Stage3Result(
        normalized_path=normalized_path,
        normalization_skipped=False,
        normalization_applied=normalization_applied,
        known_biases=known_biases,
        loudnorm_measurement=measurement,
    )