"""
src/utils/audio/profiling.py
=============================
TrustScript Phase 1 — Stage 2: Audio Quality Profiling

Responsibility
--------------
Characterize raw input audio quality before normalization, with all
measurements anchored to Signal Time (sample offsets in the original
file's sample space).

Measurements produced
---------------------
    file_level      — aggregate quality summary across the full file
    quality_windows — 30-second non-overlapping quality slices
    quality_events  — contiguous zones of degraded audio quality

Signal Time architecture
------------------------
Every measurement that references a position in the audio carries a
sample offset as its primary key:

    start_sample: int   ← primary key, integer, never accumulated
    start_seconds: float ← derived display value only

The conversion is computed once per frame from the original sample rate:

    analysis_sample  = frame_index * FRAME_SAMPLES
    original_sample  = round(analysis_sample * (original_sr / 16000))
    seconds          = original_sample / original_sr

Using integer sample arithmetic prevents IEEE 754 floating point
accumulation error. At 48kHz over 2 hours, naive floating-point
accumulation can drift by several samples — in forensic audio, the
difference between adjacent samples may have evidentiary significance.

Streaming architecture
-----------------------
Audio is read via soundfile.blocks() — one 20ms frame at a time.
Peak amplitude and per-frame RMS values accumulate in-memory during
the streaming pass. The working WAV is never loaded in full.

    Full waveform load:       ~460 MB for a 2-hour 48kHz file
    Streaming approach:       one frame (~7.5 KB stereo) at a time
                              + arrays: ~5.8 MB total for 2 hours

Two-pass approach
-----------------
1. Streaming pass: collect rms_per_frame and peak_per_frame arrays.
2. Computation pass: compute sliding-window SNR series, build quality
   windows, detect quality events. All computation uses the arrays
   from pass 1 — no second file read.

VAD method
----------
Energy-based frame-level VAD. No neural model, no HuggingFace auth.
Percentile-based thresholds are self-calibrating to each file's energy
distribution. A refined pyannote-VAD SNR is produced in Stage 5.

Contributors
------------
    Tunable constants: _SILENCE_PERCENTILE, _SPEECH_PERCENTILE,
    _NOISE_INSTABILITY_THRESHOLD, _SNR_THRESHOLDS.
    After collecting a real BWC corpus, calibrate these and document
    the dataset used.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .events import EventDetector, _SlidingWindowResult
from .models import AudioQualityProfile, Stage2Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------

#: Sample rate used for the profiling analysis pass.
#: Original sample rate is preserved from Stage 1 and written to output.
_ANALYSIS_SAMPLE_RATE: int = 16_000

#: Frame duration in milliseconds. 20ms is standard for speech processing.
_FRAME_DURATION_MS: float = 20.0

#: Samples per analysis frame at _ANALYSIS_SAMPLE_RATE.
_FRAME_SAMPLES: int = int(_ANALYSIS_SAMPLE_RATE * _FRAME_DURATION_MS / 1000.0)

#: soundfile read timeout is managed by the OS — no explicit timeout needed.


# ---------------------------------------------------------------------------
# Energy-based VAD parameters
# ---------------------------------------------------------------------------

#: Frames at or below this RMS percentile are classified as silence.
_SILENCE_PERCENTILE: float = 20.0

#: Frames at or above this RMS percentile are classified as speech.
#: Frames between the two percentiles are ambiguous and excluded.
_SPEECH_PERCENTILE: float = 50.0

#: Minimum silence frames for a reliable noise floor estimate.
_MIN_SILENCE_FRAMES: int = 10

#: Epsilon to prevent log(0) and division-by-zero.
_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Noise stationarity
# ---------------------------------------------------------------------------

#: CV above this threshold indicates a non-stationary noise floor.
_NOISE_INSTABILITY_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Sliding window parameters
# ---------------------------------------------------------------------------

#: Sliding window duration in seconds for per-window SNR computation.
_SLIDING_WINDOW_SECONDS: float = 5.0

#: Hop between sliding window positions in seconds.
_SLIDING_HOP_SECONDS: float = 1.0


# ---------------------------------------------------------------------------
# SNR classification thresholds
# ---------------------------------------------------------------------------

#: (min_snr_db_inclusive, label, snr_flagged). Evaluated top-down.
_SNR_THRESHOLDS: list[tuple[float, str, bool]] = [
    (30.0,          "EXCELLENT", False),
    (20.0,          "GOOD",      False),
    (15.0,          "MODERATE",  False),
    (10.0,          "POOR",      True),
    (float("-inf"), "CRITICAL",  True),
]

#: Linear amplitude equivalent of -1 dBFS clipping threshold.
_CLIPPING_THRESHOLD_LINEAR: float = 10 ** (-1.0 / 20.0)  # ≈ 0.8913


# ---------------------------------------------------------------------------
# Streaming pass
# ---------------------------------------------------------------------------


def _stream_frame_arrays(
    working_audio_path: Path,
    original_sample_rate: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Stream the working WAV via soundfile.blocks() and collect per-frame
    RMS and peak amplitude arrays.

    This is the single I/O pass for Stage 2. Both clipping detection and
    SNR estimation derive from the arrays returned here.

    Memory profile
    --------------
    One 20ms frame is in RAM at a time. At 48kHz stereo:
        frame shape: (960, 2) float32 ≈ 7.5 KB
    What accumulates:
        rms_per_frame:  one float32 per frame  ≈ 2.9 MB for 2 hours
        peak_per_frame: one float32 per frame  ≈ 2.9 MB for 2 hours

    Stereo handling
    ---------------
    soundfile returns shape (n_samples, n_channels) for multi-channel
    audio. Each block is averaged across channels before RMS/peak
    computation:
        block.mean(axis=1)  →  (n_samples,)

    Signal Time anchoring
    ---------------------
    The arrays are indexed by frame number. Frame i starts at:
        original_start_sample = round(
            i * FRAME_SAMPLES * (original_sr / ANALYSIS_SR)
        )
    This conversion is performed by EventDetector.frame_to_original_sample().

    Parameters
    ----------
    working_audio_path : Path
        Working PCM f32le WAV from extraction.py.
    original_sample_rate : int
        Original sample rate from Stage 1. Used to log correct duration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        (rms_per_frame, peak_per_frame) as float32 arrays.
        None if soundfile fails to open or read the file.
    """
    rms_list: list[float] = []
    peak_list: list[float] = []

    try:
        for block in sf.blocks(
            working_audio_path,
            blocksize=_FRAME_SAMPLES,
            overlap=0,
            dtype="float32",
        ):
            if block.size == 0:
                continue

            # Downmix to mono on the fly — no full-file mono array allocated.
            if block.ndim == 2:
                block = block.mean(axis=1)

            rms_list.append(float(np.sqrt(np.mean(block ** 2))))
            peak_list.append(float(np.max(np.abs(block))))

    except sf.SoundFileError as e:
        logger.warning(
            "soundfile failed to read %s: %s",
            working_audio_path.name, e,
        )
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error reading %s: %s",
            working_audio_path.name, e,
        )
        return None

    if not rms_list:
        logger.warning("No frames read from %s.", working_audio_path.name)
        return None

    rms_array = np.array(rms_list, dtype=np.float32)
    peak_array = np.array(peak_list, dtype=np.float32)

    logger.debug(
        "Streamed %s: %d frames at %d Hz (%.1f seconds)",
        working_audio_path.name,
        len(rms_list),
        _ANALYSIS_SAMPLE_RATE,
        len(rms_list) * _FRAME_DURATION_MS / 1000.0,
    )

    return rms_array, peak_array


# ---------------------------------------------------------------------------
# Clipping detection (file-level)
# ---------------------------------------------------------------------------


def _detect_clipping(peak_array: np.ndarray) -> tuple[bool, float]:
    """
    Detect digital clipping from the full-file peak amplitude array.

    Parameters
    ----------
    peak_array : np.ndarray
        Per-frame peak amplitude values from _stream_frame_arrays().

    Returns
    -------
    tuple[bool, float]
        (clipping_detected, peak_dbfs)
    """
    peak_linear = float(np.max(peak_array))
    peak_dbfs = 20.0 * np.log10(peak_linear + _EPSILON)
    clipping = peak_dbfs >= -1.0

    if clipping:
        logger.warning(
            "Clipping detected: peak %.2f dBFS (threshold: -1.0 dBFS).",
            peak_dbfs,
        )
    else:
        logger.debug("No clipping. Peak: %.2f dBFS", peak_dbfs)

    return clipping, float(peak_dbfs)


# ---------------------------------------------------------------------------
# File-level VAD thresholds
# ---------------------------------------------------------------------------


def _compute_vad_thresholds(
    rms_array: np.ndarray,
) -> tuple[float, float, float | None]:
    """
    Compute global silence/speech thresholds from the full-file RMS array.

    These thresholds are computed once from the full file and used
    consistently across all sliding windows and quality windows.
    Per-window local thresholds would be inconsistent and make
    cross-window comparisons meaningless.

    Parameters
    ----------
    rms_array : np.ndarray
        Per-frame RMS values for the full file.

    Returns
    -------
    tuple[float, float, float | None]
        (silence_threshold, speech_threshold, speech_fraction)
        speech_fraction: fraction of frames above speech_threshold.
        None if the array is too short for percentile computation.
    """
    if len(rms_array) < 10:
        return 0.0, float("inf"), None

    silence_threshold = float(np.percentile(rms_array, _SILENCE_PERCENTILE))
    speech_threshold = float(np.percentile(rms_array, _SPEECH_PERCENTILE))
    speech_fraction = float(np.mean(rms_array >= speech_threshold))

    return silence_threshold, speech_threshold, speech_fraction


# ---------------------------------------------------------------------------
# Sliding window SNR series
# ---------------------------------------------------------------------------


def _compute_sliding_window_snrs(
    rms_array: np.ndarray,
    peak_array: np.ndarray,
    silence_threshold: float,
    speech_threshold: float,
    detector: EventDetector,
) -> list[_SlidingWindowResult]:
    """
    Compute per-window SNR and stability using a sliding window over
    the per-frame RMS array.

    Window and hop are expressed in frames (derived from seconds), not
    seconds directly — this avoids floating point accumulation in the
    window positioning loop.

    Signal Time anchors
    -------------------
    Each window's start_sample and end_sample are computed by calling
    detector.frame_to_original_sample(), which uses integer arithmetic
    in the original sample space. start_seconds and end_seconds are
    derived display values only.

    Parameters
    ----------
    rms_array : np.ndarray
        Per-frame RMS values from the streaming pass.
    peak_array : np.ndarray
        Per-frame peak amplitude values from the streaming pass.
    silence_threshold : float
        Global silence threshold from _compute_vad_thresholds().
    speech_threshold : float
        Global speech threshold from _compute_vad_thresholds().
    detector : EventDetector
        For Signal Time conversions.

    Returns
    -------
    list[_SlidingWindowResult]
        One entry per sliding window position, ordered by start_frame.
    """
    window_frames = int(_SLIDING_WINDOW_SECONDS * 1000 / _FRAME_DURATION_MS)
    hop_frames = int(_SLIDING_HOP_SECONDS * 1000 / _FRAME_DURATION_MS)
    n_frames = len(rms_array)
    results: list[_SlidingWindowResult] = []

    for start_frame in range(0, n_frames - window_frames + 1, hop_frames):
        end_frame = start_frame + window_frames
        window_rms = rms_array[start_frame:end_frame]
        window_peak = peak_array[start_frame:end_frame]

        # Signal Time anchors — integer sample offsets, not accumulated floats.
        start_sample = detector.frame_to_original_sample(start_frame)
        end_sample = detector.frame_to_original_sample(end_frame)
        start_seconds = start_sample / detector.original_sr
        end_seconds = end_sample / detector.original_sr

        # Local speech/silence using global thresholds.
        silence_frames = window_rms[window_rms <= silence_threshold]
        speech_frames = window_rms[window_rms >= speech_threshold]
        speech_fraction = float(len(speech_frames)) / len(window_rms)

        # Local SNR.
        snr_db: float | None = None
        if len(silence_frames) >= _MIN_SILENCE_FRAMES and len(speech_frames) > 0:
            mean_speech = float(np.mean(speech_frames))
            mean_silence = float(np.mean(silence_frames))
            if mean_silence > _EPSILON:
                snr_db = float(20.0 * np.log10(mean_speech / mean_silence))

        # Local noise stability (CV).
        noise_cv: float | None = None
        if len(silence_frames) >= _MIN_SILENCE_FRAMES:
            mean_sil = float(np.mean(silence_frames))
            std_sil = float(np.std(silence_frames))
            noise_cv = std_sil / (mean_sil + _EPSILON)

        # Clipping in this window.
        clipping_detected = bool(np.any(window_peak >= _CLIPPING_THRESHOLD_LINEAR))

        results.append(_SlidingWindowResult(
            start_frame=start_frame,
            end_frame=end_frame,
            start_sample=start_sample,
            end_sample=end_sample,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            snr_db=snr_db,
            noise_cv=noise_cv,
            speech_fraction=speech_fraction,
            clipping_detected=clipping_detected,
        ))

    logger.debug(
        "Computed %d sliding window SNR values (%.0fs window, %.0fs hop)",
        len(results),
        _SLIDING_WINDOW_SECONDS,
        _SLIDING_HOP_SECONDS,
    )

    return results


# ---------------------------------------------------------------------------
# File-level SNR and stability aggregation
# ---------------------------------------------------------------------------


def _aggregate_file_level_snr(
    rms_array: np.ndarray,
    silence_threshold: float,
    speech_threshold: float,
) -> tuple[float | None, float | None, bool, str | None]:
    """
    Compute file-level SNR and noise stability from the full RMS array.

    Used to populate the file_level block in the output JSON. The sliding
    window series gives time-indexed precision; this gives an overall summary.

    Returns
    -------
    tuple[float|None, float|None, bool, str|None]
        (snr_db, noise_cv, noise_is_unstable, note)
    """
    silence_rms = rms_array[rms_array <= silence_threshold]
    speech_rms = rms_array[rms_array >= speech_threshold]

    if len(silence_rms) < _MIN_SILENCE_FRAMES:
        note = (
            f"Only {len(silence_rms)} silence frame(s) — "
            "audio may be continuous speech. SNR estimate unreliable."
        )
        return None, None, False, note

    if len(speech_rms) == 0:
        note = "No speech frames detected. Audio may be near-silent."
        return None, None, False, note

    mean_speech = float(np.mean(speech_rms))
    mean_silence = float(np.mean(silence_rms))

    if mean_silence < _EPSILON:
        return 60.0, 0.0, False, "Noise floor near zero — SNR capped at 60 dB."

    snr_db = float(20.0 * np.log10(mean_speech / mean_silence))

    # Noise stability.
    noise_cv = float(np.std(silence_rms) / (mean_silence + _EPSILON))
    noise_is_unstable = noise_cv > _NOISE_INSTABILITY_THRESHOLD

    note: str | None = None
    if noise_is_unstable:
        note = (
            f"Non-stationary noise floor (CV={noise_cv:.3f} > "
            f"{_NOISE_INSTABILITY_THRESHOLD}). "
            f"SNR of {snr_db:.1f} dB may be over-optimistic. "
            "See quality_events for time-indexed degraded zones."
        )

    return snr_db, noise_cv, noise_is_unstable, note


def _classify_snr(snr_db: float | None) -> tuple[str, bool]:
    """Map SNR to classification label and flag status."""
    if snr_db is None:
        return "UNKNOWN", True
    for min_snr, label, flagged in _SNR_THRESHOLDS:
        if snr_db >= min_snr:
            return label, flagged
    return "UNKNOWN", True


# ---------------------------------------------------------------------------
# Stage 2 entry point
# ---------------------------------------------------------------------------


def run_stage2(
    stage1_result: "Stage1Result",  # type: ignore[name-defined]
    working_audio_path: Path,
) -> Stage2Result:
    """
    Execute Stage 2: Audio Quality Profiling.

    Two-pass approach:
        Pass 1 (I/O):        Stream the working WAV frame-by-frame via
                             soundfile.blocks(). Collect rms_per_frame
                             and peak_per_frame arrays.
        Pass 2 (Computation): Compute sliding window SNR series, build
                             30s quality windows, detect quality events.
                             No second file read.

    All time-referenced measurements carry sample offsets (original
    sample space) as primary Signal Time keys.

    Parameters
    ----------
    stage1_result : Stage1Result
        Complete output from run_stage1().
    working_audio_path : Path
        Working PCM WAV produced by extract_working_audio().

    Returns
    -------
    Stage2Result
        Complete Stage 2 profiling results. Pass to run_stage3().
    """
    from .stage1 import Stage1Result  # Avoid circular import.

    path = stage1_result.original_path
    props = stage1_result.audio_properties

    logger.info("Stage 2 — Audio Quality Profiling: %s", path.name)
    logger.debug(
        "Working audio: %s | original SR: %d Hz",
        working_audio_path, props.sample_rate,
    )

    # Instantiate EventDetector with the scale factor for this file.
    detector = EventDetector(
        original_sample_rate=props.sample_rate,
        analysis_sample_rate=_ANALYSIS_SAMPLE_RATE,
        frame_samples=_FRAME_SAMPLES,
    )

    # ------------------------------------------------------------------
    # Pass 1: Stream the working WAV, collect per-frame arrays.
    # ------------------------------------------------------------------
    stream_result = _stream_frame_arrays(working_audio_path, props.sample_rate)

    if stream_result is None:
        logger.warning(
            "Streaming pass failed for %s — "
            "all quality measurements unavailable.",
            path.name,
        )
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
            snr_note="Waveform streaming failed — measurements unavailable.",
        )
        return Stage2Result(audio_quality=audio_quality)

    rms_array, peak_array = stream_result
    n_frames = len(rms_array)

    # ------------------------------------------------------------------
    # Signal Time integrity check — drift detection.
    # Compare actual frames processed against the expected total_samples
    # from Stage 1 ffprobe. A discrepancy beyond one frame's worth of
    # samples indicates timestamp drift or a truncated/corrupt file.
    # ------------------------------------------------------------------
    actual_samples_processed = n_frames * _FRAME_SAMPLES
    expected_samples = props.total_samples
    sample_tolerance = _FRAME_SAMPLES  # one frame of acceptable rounding

    if (
        expected_samples > 0
        and abs(actual_samples_processed - expected_samples) > sample_tolerance
    ):
        logger.warning(
            "Signal Time drift detected for %s: "
            "processed %d samples, expected %d samples (delta: %d). "
            "File may be truncated or corrupt. "
            "Quality window timestamps may not align with the full recording.",
            path.name,
            actual_samples_processed,
            expected_samples,
            actual_samples_processed - expected_samples,
        )
    else:
        logger.debug(
            "Signal Time integrity check passed: "
            "processed %d samples, expected %d (delta: %d).",
            actual_samples_processed,
            expected_samples,
            actual_samples_processed - expected_samples,
        )

    # ------------------------------------------------------------------
    # Pass 2: All computation from the in-memory arrays.
    # ------------------------------------------------------------------

    # Clipping detection — file-level.
    clipping_detected, peak_dbfs = _detect_clipping(peak_array)

    # Global VAD thresholds from full-file RMS distribution.
    silence_threshold, speech_threshold, speech_fraction = (
        _compute_vad_thresholds(rms_array)
    )

    # Energy percentile summary for output metadata.
    pct_keys = [10, 20, 50, 80, 90]
    energy_percentiles = {
        f"p{p}": float(np.percentile(rms_array, p))
        for p in pct_keys
    }

    # File-level SNR and stability.
    snr_db, noise_cv, noise_is_unstable, snr_note = _aggregate_file_level_snr(
        rms_array, silence_threshold, speech_threshold,
    )
    snr_classification, snr_flagged = _classify_snr(snr_db)

    if snr_flagged and snr_db is not None:
        logger.warning(
            "Low SNR: %.1f dB (%s) for %s.",
            snr_db, snr_classification, path.name,
        )

    # Sliding window SNR series — Signal Time anchored.
    sliding_results = _compute_sliding_window_snrs(
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