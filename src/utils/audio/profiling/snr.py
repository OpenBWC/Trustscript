"""
src/utils/audio/snr.py
=======================
TrustScript Phase 1 — Stage 2 SNR Computation

Overhaul Coming: I will be pitvoting away from the average methodology soon

Responsibility
--------------
All Signal-to-Noise Ratio computation for Stage 2:

    1. File-level SNR aggregation
       Computes a single SNR estimate for the full file using global
       VAD thresholds from vad.py.

    2. Sliding window SNR series
       Computes local SNR at each position of a 5-second sliding window
       with 1-second hop, producing a time-indexed SNR series that feeds
       EventDetector in events.py.

    3. SNR classification
       Maps an SNR value to a human-readable quality label
       (EXCELLENT / GOOD / MODERATE / POOR / CRITICAL / UNKNOWN).

SNR formula
-----------
    SNR (dB) = 20 × log10(mean_speech_rms / mean_silence_rms)

This requires two RMS measurements: one from speech-active frames and
one from silence frames. The VAD classification (from vad.py) provides
the frame labels.

Signal Time in sliding windows
--------------------------------
Each sliding window result carries ``start_sample`` and ``end_sample``
in the original file's sample space, computed by
``EventDetector.frame_to_original_sample()``. Seconds values are derived
display values only. See events.py for the full Signal Time contract.

Contributors
------------
    Do not add I/O, VAD threshold computation, or event detection logic
    here. This module owns only SNR computation and classification.
    If you adjust _SNR_THRESHOLDS, update constants.py — the thresholds
    are defined there as SNR_THRESHOLDS and imported here.
"""

import logging

import numpy as np

from .constants import (
    CLIPPING_THRESHOLD_LINEAR,
    EPSILON,
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    MIN_SILENCE_FRAMES,
    MIN_WINDOW_FRAMES,
    NOISE_INSTABILITY_THRESHOLD,
    SLIDING_HOP_SECONDS,
    SLIDING_WINDOW_SECONDS,
    SNR_THRESHOLDS,
)
from .events import _SlidingWindowResult
from .vad import check_noise_stability

logger = logging.getLogger(__name__)


def classify_snr(snr_db: float | None) -> tuple[str, bool]:
    """
    Map an SNR value to a quality classification label and flag status.

    Parameters
    ----------
    snr_db : float | None
        Estimated SNR in dB. Pass ``None`` when SNR could not be computed.

    Returns
    -------
    tuple[str, bool]
        ``(classification_label, snr_flagged)``
        ``classification_label``: one of EXCELLENT, GOOD, MODERATE,
        POOR, CRITICAL, UNKNOWN.
        ``snr_flagged``: ``True`` for POOR, CRITICAL, and UNKNOWN.
    """
    if snr_db is None:
        return "UNKNOWN", True

    for min_snr, label, flagged in SNR_THRESHOLDS:
        if snr_db >= min_snr:
            return label, flagged

    return "UNKNOWN", True  # Unreachable given SNR_THRESHOLDS covers -inf.


def compute_file_level_snr(
    rms_array: np.ndarray,
    silence_threshold: float,
    speech_threshold: float,
) -> tuple[float | None, float | None, bool, str | None]:
    """
    Compute file-level SNR and noise stability from the full RMS array.

    This produces the aggregate ``file_level`` summary block. The sliding
    window series from ``compute_sliding_window_snrs()`` provides the
    time-indexed precision; this gives the overall summary for quick triage.

    Parameters
    ----------
    rms_array : np.ndarray
        Per-frame RMS values for the full file, from streaming.py.
    silence_threshold : float
        Global silence threshold from ``vad.compute_vad_thresholds()``.
    speech_threshold : float
        Global speech threshold from ``vad.compute_vad_thresholds()``.

    Returns
    -------
    tuple[float | None, float | None, bool, str | None]
        ``(snr_db, noise_cv, noise_is_unstable, note)``
        ``note``: human-readable caveat, or ``None`` if computation
        was clean.
    """
    silence_rms = rms_array[rms_array <= silence_threshold]
    speech_rms = rms_array[rms_array >= speech_threshold]

    if len(silence_rms) < MIN_SILENCE_FRAMES:
        note = (
            f"Only {len(silence_rms)} silence frame(s) found "
            f"(minimum: {MIN_SILENCE_FRAMES}). "
            "Audio may be continuous speech — SNR estimate unreliable."
        )
        logger.warning("File-level SNR skipped: %s", note)
        return None, None, False, note

    if len(speech_rms) == 0:
        note = (
            "No speech frames detected above the energy threshold. "
            "Audio may be near-silent or contain only background noise."
        )
        logger.warning("File-level SNR skipped: %s", note)
        return None, None, False, note

    mean_speech = float(np.mean(speech_rms))
    mean_silence = float(np.mean(silence_rms))

    if mean_silence < EPSILON:
        note = (
            "Noise floor is effectively zero — audio may have been "
            "pre-processed to remove silence. SNR capped at 60 dB."
        )
        logger.debug(note)
        return 60.0, 0.0, False, note

    snr_db = float(20.0 * np.log10(mean_speech / mean_silence))

    # Noise floor stationarity check.
    # Applied after MIN_SILENCE_FRAMES guard — std() is meaningless
    # on very small samples.
    noise_cv, noise_is_unstable = check_noise_stability(silence_rms)

    note: str | None = None
    if noise_is_unstable:
        note = (
            f"Non-stationary noise floor (CV={noise_cv:.3f} > "
            f"{NOISE_INSTABILITY_THRESHOLD}). "
            f"SNR of {snr_db:.1f} dB may be over-optimistic. "
            "See quality_events for time-indexed degraded zones."
        )

    logger.debug(
        "File-level SNR: %.1f dB | noise CV: %s",
        snr_db,
        f"{noise_cv:.3f}" if noise_cv is not None else "n/a",
    )

    return snr_db, noise_cv, noise_is_unstable, note


def compute_sliding_window_snrs(
    rms_array: np.ndarray,
    peak_array: np.ndarray,
    silence_threshold: float,
    speech_threshold: float,
    detector: "EventDetector",  # type: ignore[name-defined]
) -> list[_SlidingWindowResult]:
    """
    Compute per-window SNR and noise stability using a sliding window.

    Window and hop are expressed in frames (derived from seconds) to avoid
    floating point accumulation in the window positioning loop.

    Signal Time anchors
    -------------------
    Each window's ``start_sample`` and ``end_sample`` are computed by
    ``detector.frame_to_original_sample()``, which uses integer arithmetic
    in the original sample space. ``start_seconds`` and ``end_seconds``
    are derived display values only — never used as keys.

    Parameters
    ----------
    rms_array : np.ndarray
        Per-frame RMS values from streaming.py.
    peak_array : np.ndarray
        Per-frame peak amplitude values from streaming.py.
    silence_threshold : float
        Global silence threshold from ``vad.compute_vad_thresholds()``.
    speech_threshold : float
        Global speech threshold from ``vad.compute_vad_thresholds()``.
    detector : EventDetector
        For Signal Time conversions. Import from events.py.

    Returns
    -------
    list[_SlidingWindowResult]
        One entry per sliding window position, ordered by ``start_frame``
        ascending. Fed directly to ``EventDetector.detect_quality_events()``.
    """
    # Express window and hop in frames — never in accumulated seconds.
    window_frames = int(SLIDING_WINDOW_SECONDS * 1000 / FRAME_DURATION_MS)
    hop_frames = int(SLIDING_HOP_SECONDS * 1000 / FRAME_DURATION_MS)
    n_frames = len(rms_array)
    results: list[_SlidingWindowResult] = []

    for start_frame in range(0, n_frames - window_frames + 1, hop_frames):
        end_frame = start_frame + window_frames
        window_rms = rms_array[start_frame:end_frame]
        window_peak = peak_array[start_frame:end_frame]

        # Signal Time anchors — integer arithmetic, not accumulated floats.
        start_sample = detector.frame_to_original_sample(start_frame)
        end_sample = detector.frame_to_original_sample(end_frame)
        start_seconds = start_sample / detector.original_sr
        end_seconds = end_sample / detector.original_sr

        # Local speech/silence classification using global thresholds.
        silence_frames = window_rms[window_rms <= silence_threshold]
        speech_frames = window_rms[window_rms >= speech_threshold]
        speech_fraction = float(len(speech_frames)) / len(window_rms)

        # Local SNR.
        snr_db: float | None = None
        if (
            len(silence_frames) >= MIN_SILENCE_FRAMES
            and len(speech_frames) > 0
        ):
            mean_speech = float(np.mean(speech_frames))
            mean_silence = float(np.mean(silence_frames))
            if mean_silence > EPSILON:
                snr_db = float(20.0 * np.log10(mean_speech / mean_silence))

        # Local noise stability (CV).
        noise_cv: float | None = None
        if len(silence_frames) >= MIN_SILENCE_FRAMES:
            noise_cv, _ = check_noise_stability(silence_frames)

        # Clipping in this window.
        from .constants import CLIPPING_THRESHOLD_LINEAR
        clipping_detected = bool(
            np.any(window_peak >= CLIPPING_THRESHOLD_LINEAR)
        )

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
        "Computed %d sliding window SNR values (%.0fs window, %.0fs hop).",
        len(results),
        SLIDING_WINDOW_SECONDS,
        SLIDING_HOP_SECONDS,
    )

    return results