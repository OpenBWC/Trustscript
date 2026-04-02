"""
src/utils/audio/vad.py
=======================
TrustScript Phase 1 — Stage 2 Voice Activity Detection and Clipping

Responsibility
--------------
Two signal characterisation operations that operate on the per-frame
arrays produced by streaming.py:

    1. VAD threshold computation
       Derives global silence and speech RMS thresholds from the full-file
       frame energy distribution using percentile-based classification.

    2. Clipping detection
       Determines whether any frame's peak amplitude exceeded -1 dBFS.

Neither function reads audio files or performs SNR computation.
SNR computation belongs in snr.py.

Why percentile-based VAD?
--------------------------
BWC audio has a highly variable noise floor — wind, sirens, and radio
chatter shift the absolute energy level throughout the recording.
A fixed amplitude threshold would misclassify loud noise as speech or
quiet speech as silence. Percentile-based classification is self-
calibrating to each file's own energy distribution, making it more
robust to the conditions TrustScript is designed for.

These thresholds are computed once from the full file and used
consistently across all sliding windows and quality windows in snr.py
and events.py. Per-window local thresholds would make cross-window
SNR comparisons inconsistent and meaningless.

Contributors
------------
    Do not add SNR computation or event detection logic here. This module
    owns only VAD threshold derivation and clipping detection.
"""

import logging

import numpy as np

from .constants import (
    CLIPPING_THRESHOLD_LINEAR,
    EPSILON,
    MIN_SILENCE_FRAMES,
    NOISE_INSTABILITY_THRESHOLD,
    SILENCE_PERCENTILE,
    SPEECH_PERCENTILE,
)

logger = logging.getLogger(__name__)


def compute_vad_thresholds(
    rms_array: np.ndarray,
) -> tuple[float, float, float | None]:
    """
    Compute global silence/speech RMS thresholds from the full-file array.

    Frames at or below the silence threshold are classified as silence.
    Frames at or above the speech threshold are classified as speech.
    Frames between the two thresholds are "ambiguous" and excluded from
    SNR computation.

    Parameters
    ----------
    rms_array : np.ndarray
        Per-frame RMS values for the full file, from streaming.py.

    Returns
    -------
    tuple[float, float, float | None]
        ``(silence_threshold, speech_threshold, speech_fraction)``
        ``speech_fraction``: fraction of frames above speech_threshold.
        Returns ``(0.0, inf, None)`` if the array is too short for
        meaningful percentile computation.
    """
    if len(rms_array) < 10:
        logger.warning(
            "RMS array has only %d frames — too short for VAD threshold "
            "computation. Returning degenerate thresholds.",
            len(rms_array),
        )
        return 0.0, float("inf"), None

    silence_threshold = float(np.percentile(rms_array, SILENCE_PERCENTILE))
    speech_threshold = float(np.percentile(rms_array, SPEECH_PERCENTILE))
    speech_fraction = float(np.mean(rms_array >= speech_threshold))

    logger.debug(
        "VAD thresholds: silence ≤ %.6f RMS (p%d), "
        "speech ≥ %.6f RMS (p%d), speech fraction: %.2f",
        silence_threshold, int(SILENCE_PERCENTILE),
        speech_threshold, int(SPEECH_PERCENTILE),
        speech_fraction,
    )

    return silence_threshold, speech_threshold, speech_fraction


def detect_clipping(peak_array: np.ndarray) -> tuple[bool, float]:
    """
    Detect digital clipping from the full-file peak amplitude array.

    Digital clipping occurs when the recorded signal exceeds the maximum
    representable level. In normalised float32 audio, 0 dBFS = 1.0.
    The threshold of -1 dBFS (≈ 0.891 linear) provides a small margin
    that catches samples approaching saturation as well as fully clipped
    samples.

    Clipping cannot be recovered through normalisation — it permanently
    distorts the waveform. The presence of clipping in the original
    recording is documented in the output metadata for chain-of-custody
    purposes.

    Parameters
    ----------
    peak_array : np.ndarray
        Per-frame peak amplitude values from streaming.py.
        Values are absolute amplitudes in ``[0.0, ∞)`` for pcm_f32le
        (may exceed 1.0 for samples that were above 0 dBFS in the source).

    Returns
    -------
    tuple[bool, float]
        ``(clipping_detected, peak_dbfs)``
        ``peak_dbfs``: maximum peak across all frames in dBFS.
        Values ≥ -1.0 indicate clipping.
    """
    peak_linear = float(np.max(peak_array))
    peak_dbfs = 20.0 * np.log10(peak_linear + EPSILON)
    clipping = peak_dbfs >= -1.0

    if clipping:
        logger.warning(
            "Clipping detected: peak %.2f dBFS (threshold: -1.0 dBFS). "
            "The recording was captured at too high a level. "
            "Clipping cannot be recovered through normalisation.",
            peak_dbfs,
        )
    else:
        logger.debug("No clipping. Peak amplitude: %.2f dBFS", peak_dbfs)

    return clipping, float(peak_dbfs)


def check_noise_stability(
    silence_rms: np.ndarray,
) -> tuple[float | None, bool]:
    """
    Measure whether the detected noise floor is stationary using the
    Coefficient of Variation (CV).

    CV = std(silence_rms) / mean(silence_rms)

    A low CV indicates a flat, consistent noise floor (stationary).
    A high CV indicates a variable, chaotic noise floor (non-stationary),
    meaning the SNR estimate may be over-optimistic — the "silence"
    frames contain energy bursts rather than a true background level.

    This check must be called after the ``MIN_SILENCE_FRAMES`` guard.
    With fewer frames, ``std()`` is statistically meaningless.

    Parameters
    ----------
    silence_rms : np.ndarray
        RMS values of frames classified as silence by ``compute_vad_thresholds()``.
        Must have ``len >= MIN_SILENCE_FRAMES`` (caller's responsibility).

    Returns
    -------
    tuple[float | None, bool]
        ``(noise_cv, is_unstable)``
        ``noise_cv``: computed CV. ``None`` if the array is empty.
        ``is_unstable``: ``True`` when ``noise_cv > NOISE_INSTABILITY_THRESHOLD``.
    """
    if len(silence_rms) == 0:
        return None, True

    mean_silence = float(np.mean(silence_rms))
    std_silence = float(np.std(silence_rms))
    noise_cv = std_silence / (mean_silence + EPSILON)
    is_unstable = noise_cv > NOISE_INSTABILITY_THRESHOLD

    if is_unstable:
        logger.warning(
            "Unstable noise floor: CV=%.3f (threshold: %.2f). "
            "Silence frames show high energy variation — possible rain, "
            "wind gusts, or intermittent radio. SNR may be over-optimistic.",
            noise_cv, NOISE_INSTABILITY_THRESHOLD,
        )
    else:
        logger.debug(
            "Noise floor stable: CV=%.3f (threshold: %.2f).",
            noise_cv, NOISE_INSTABILITY_THRESHOLD,
        )

    return float(noise_cv), is_unstable