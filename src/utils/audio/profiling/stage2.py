"""
src/utils/audio/stage2.py
==========================
TrustScript Phase 1 — Stage 2 Orchestrator

Responsibility
--------------
Coordinate the Stage 2: Audio Quality Profiling pipeline by calling
each sub-module in sequence and assembling the result. This module
contains only the ``run_stage2()`` entry point — no computation logic.

Stage 2 sub-modules
--------------------
    streaming.py   Pass 1 (I/O): stream frames, collect RMS + peak arrays
    vad.py         Global VAD thresholds + clipping detection
    snr.py         File-level SNR + sliding window SNR series
    events.py      EventDetector: quality windows + quality events
    constants.py   All tunable parameters

Two-pass architecture
---------------------
    Pass 1 (I/O):        ``streaming.stream_frame_arrays()``
                         Streams the working WAV frame-by-frame.
                         Collects ``rms_per_frame`` and ``peak_per_frame``.

    Pass 2 (Computation): All subsequent calls operate on the in-memory
                          arrays from Pass 1. No second file read.

Signal Time contract
--------------------
All time-referenced measurements carry sample offsets in the original
file's sample space as primary keys. Seconds values are derived display
values only. See events.py for the full contract and conversion logic.

Contributors
------------
    Do not add computation logic to this module. If you need a new
    Stage 2 measurement, add it to the appropriate sub-module and call
    it from ``_run_diarize_pipeline`` in diarize.py, or wire it in here.
"""

import logging
from pathlib import Path

import numpy as np

from .constants import FRAME_SAMPLES, QUALITY_WINDOW_SECONDS
from .events import EventDetector
from ..models import AudioQualityProfile, Stage2Result
from .snr import classify_snr, compute_file_level_snr, compute_sliding_window_snrs
from .streaming import stream_frame_arrays
from .vad import compute_vad_thresholds, detect_clipping

logger = logging.getLogger(__name__)

#: Number of samples tolerance for Signal Time drift detection.
#: One frame's worth of samples accounts for the rounding on the
#: final partial frame at the end of the file.
_DRIFT_TOLERANCE_FRAMES: int = 1


def run_stage2(
    stage1_result: "Stage1Result",  # type: ignore[name-defined]
    working_audio_path: Path,
) -> Stage2Result:
    """
    Execute Stage 2: Audio Quality Profiling.

    Orchestrates the streaming pass, VAD threshold computation,
    clipping detection, SNR computation, sliding window analysis,
    quality window construction, and event detection.

    Parameters
    ----------
    stage1_result : Stage1Result
        Complete output from ``run_stage1()``. Provides original audio
        properties (sample rate, channels, duration, total_samples).
    working_audio_path : Path
        Working PCM f32le WAV produced by ``extract_working_audio()``.
        May be the original file for soundfile-native formats, or an
        extracted WAV for video containers and compressed audio.

    Returns
    -------
    Stage2Result
        Complete Stage 2 profiling results. Pass to ``run_stage3()``.
    """
    from .stage1 import Stage1Result  # Avoid circular import at module load.

    path = stage1_result.original_path
    props = stage1_result.audio_properties

    logger.info("Stage 2 — Audio Quality Profiling: %s", path.name)
    logger.debug(
        "Working audio: %s | original SR: %d Hz | expected samples: %d",
        working_audio_path, props.sample_rate, props.total_samples,
    )

    # Instantiate EventDetector with the scale factor for this file.
    # Holds the integer conversion: analysis frame → original sample offset.
    detector = EventDetector(
        original_sample_rate=props.sample_rate,
        analysis_sample_rate=16_000,
        frame_samples=FRAME_SAMPLES,
    )

    # ------------------------------------------------------------------
    # Pass 1 (I/O): Stream the working WAV, collect per-frame arrays.
    # ------------------------------------------------------------------
    stream_result = stream_frame_arrays(working_audio_path, props.sample_rate)

    if stream_result is None:
        logger.warning(
            "Streaming pass failed for %s — all quality measurements unavailable.",
            path.name,
        )
        return _build_unavailable_result(props)

    rms_array, peak_array = stream_result
    n_frames = len(rms_array)

    # ------------------------------------------------------------------
    # Signal Time integrity check.
    # Compare actual samples processed against Stage 1 total_samples.
    # A discrepancy beyond one frame indicates truncation or corruption.
    # ------------------------------------------------------------------
    _check_signal_time_integrity(
        path.name, n_frames, props.total_samples
    )

    # ------------------------------------------------------------------
    # Pass 2 (Computation): All derived measurements from the arrays.
    # ------------------------------------------------------------------

    # Global VAD thresholds — computed once, used consistently everywhere.
    silence_threshold, speech_threshold, speech_fraction = (
        compute_vad_thresholds(rms_array)
    )

    # Clipping detection.
    clipping_detected, peak_dbfs = detect_clipping(peak_array)

    # Energy percentile summary for output metadata.
    energy_percentiles = {
        f"p{p}": float(np.percentile(rms_array, p))
        for p in [10, 20, 50, 80, 90]
    }

    # File-level SNR and noise stability.
    snr_db, noise_cv, noise_is_unstable, snr_note = compute_file_level_snr(
        rms_array, silence_threshold, speech_threshold,
    )
    snr_classification, snr_flagged = classify_snr(snr_db)

    if snr_flagged and snr_db is not None:
        logger.warning(
            "Low SNR: %.1f dB (%s) for %s.",
            snr_db, snr_classification, path.name,
        )

    # Sliding window SNR series — feeds event detection.
    sliding_results = compute_sliding_window_snrs(
        rms_array, peak_array,
        silence_threshold, speech_threshold,
        detector,
    )

    # 30-second quality windows — Signal Time anchored.
    quality_windows = detector.build_quality_windows(
        rms_array, peak_array,
        silence_threshold, speech_threshold,
    )

    # Quality events — contiguous degraded zones, Signal Time anchored.
    quality_events = detector.detect_quality_events(sliding_results)

    # ------------------------------------------------------------------
    # Assemble result.
    # ------------------------------------------------------------------
    audio_quality = AudioQualityProfile(
        duration_seconds=props.duration_seconds,
        sample_rate_original=props.sample_rate,
        channels_original=props.channels,
        clipping_detected=clipping_detected,
        clipping_peak_dbfs=peak_dbfs,
        snr_db=snr_db,
        snr_classification=snr_classification,
        snr_flagged=snr_flagged,
        noise_stability_cv=noise_cv,
        noise_is_unstable=noise_is_unstable,
        vad_speech_fraction=speech_fraction,
        quality_windows=quality_windows,
        quality_events=quality_events,
        vad_energy_percentiles=energy_percentiles,
        snr_note=snr_note,
    )

    logger.info(
        "Stage 2 complete — SNR: %s (%s) | noise_unstable: %s | "
        "clipping: %s | windows: %d | events: %d",
        f"{snr_db:.1f} dB" if snr_db is not None else "unknown",
        snr_classification,
        noise_is_unstable,
        clipping_detected,
        len(quality_windows),
        len(quality_events),
    )

    return Stage2Result(audio_quality=audio_quality)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_signal_time_integrity(
    filename: str,
    n_frames: int,
    expected_total_samples: int,
) -> None:
    """
    Compare actual frames processed against Stage 1 total_samples.

    Logs a WARNING if the discrepancy exceeds one frame's tolerance.
    A meaningful discrepancy indicates the file was truncated, the
    duration reported by ffprobe was inaccurate, or the streaming
    pass missed frames.
    """
    actual_samples = n_frames * FRAME_SAMPLES
    tolerance = FRAME_SAMPLES * _DRIFT_TOLERANCE_FRAMES

    if expected_total_samples <= 0:
        logger.debug(
            "Signal Time integrity check skipped — "
            "total_samples not available for %s.",
            filename,
        )
        return

    delta = actual_samples - expected_total_samples

    if abs(delta) > tolerance:
        logger.warning(
            "Signal Time integrity check FAILED for %s: "
            "processed %d samples, expected %d (delta: %+d, tolerance: ±%d). "
            "File may be truncated or ffprobe duration was inaccurate. "
            "Quality window timestamps may not cover the full recording.",
            filename,
            actual_samples,
            expected_total_samples,
            delta,
            tolerance,
        )
    else:
        logger.debug(
            "Signal Time integrity check passed for %s: "
            "processed %d samples, expected %d (delta: %+d).",
            filename,
            actual_samples,
            expected_total_samples,
            delta,
        )


def _build_unavailable_result(props: "AudioProperties") -> Stage2Result:  # type: ignore[name-defined]
    """
    Build a Stage2Result with all measurements set to unavailable.
    Used when the streaming pass fails entirely.
    """
    from .models import AudioQualityProfile, Stage2Result

    audio_quality = AudioQualityProfile(
        duration_seconds=props.duration_seconds,
        sample_rate_original=props.sample_rate,
        channels_original=props.channels,
        clipping_detected=False,
        clipping_peak_dbfs=None,
        snr_db=None,
        snr_classification="UNKNOWN",
        snr_flagged=True,
        noise_stability_cv=None,
        noise_is_unstable=False,
        vad_speech_fraction=None,
        quality_windows=[],
        quality_events=[],
        vad_energy_percentiles={},
        snr_note="Waveform streaming failed — all measurements unavailable.",
    )
    return Stage2Result(audio_quality=audio_quality)