"""
src/utils/audio/profiling.py
=============================
TrustScript Phase 1 — Stage 2: Audio Quality Profiling

Responsibility
--------------
Characterize the raw input file's audio quality before normalization.
All measurements reflect true recording conditions, not processed output.
This module is strictly read-only — no files are written or modified.

Reads from the working audio path produced by extraction.py, which is
a PCM WAV at the original sample rate and channel count.

Measurements produced
---------------------
    - Duration, original sample rate, original channel count
      (sourced from Stage 1 ffprobe output — no re-probe needed)
    - Peak amplitude and clipping detection (peak > -1 dBFS)
    - Estimated SNR via energy-based Voice Activity Detection (VAD)
    - Per-frame RMS energy percentile summary (written to output JSON
      as diagnostic evidence for SNR classification)

Memory design: soundfile.blocks() streaming
--------------------------------------------
Stage 2 uses soundfile.blocks() to stream audio frame-by-frame rather
than loading the full waveform into memory.

    Full waveform load:      ~460 MB for a 2-hour 48kHz file
    soundfile.blocks():      one 20ms frame at a time (~18 KB at 48kHz)
                             + RMS list (~2.9 MB for 2 hours)

soundfile is purpose-built for audio I/O: faster than ffmpeg for WAV
reads, better Python integration, and cleaner EOF handling. It is used
here instead of the previous ffmpeg Popen streaming approach now that
the extraction step guarantees a native WAV on disk.

Stereo handling
---------------
soundfile.blocks() returns a 2D array of shape (n_samples, n_channels)
for stereo files. RMS computation requires a 1D array. Stereo blocks
are downmixed to mono on the fly before each RMS calculation:

    block = block.mean(axis=1)  # (n, 2) → (n,)

This is a per-frame operation on a ~18KB array — no full-file downmix
is held in memory.

Non-overlapping frames
-----------------------
Stage 2 uses non-overlapping 20ms frames (blocksize = frame_samples,
no overlap). For statistical profiling — building an RMS histogram to
locate the noise floor — 50 frames per second provides more than
sufficient density. The previous 10ms hop (50% overlap) was unnecessary
for this use case and doubled the number of frames to store.

VAD method — energy-based (not pyannote)
-----------------------------------------
pyannote model loading is Stage 4. Requiring neural model downloads
for a quality profiling pass would:
    1. Block Stage 2 on HuggingFace authentication.
    2. Pay the model loading cost twice per pipeline run.
    3. Break Stage 2's "lightweight, read-only" design contract.

Energy-based VAD requires no model, no auth, and minimal RAM. It
reliably distinguishes clean from noisy recordings for the purpose of
snr_flagged classification.

A refined SNR using pyannote's actual VAD output is computed in Stage 5.

SNR reference thresholds (starting hypotheses)
-----------------------------------------------
Calibrate by plotting SNR vs. reid_score across a real BWC corpus.

    > 30 dB  EXCELLENT   Clean indoor, close mic
    20-30 dB GOOD        Normal outdoor, light wind
    15-20 dB MODERATE    Active scene, siren/radio
    10-15 dB POOR        Heavy wind, multiple sirens
    < 10 dB  CRITICAL    Near-unusable — flag for human review

Contributors
------------
    VAD parameters (_SILENCE_PERCENTILE, _SPEECH_PERCENTILE) and SNR
    thresholds (_SNR_THRESHOLDS) are tunable. After collecting a corpus
    of real BWC files, update thresholds with empirically calibrated
    values and document the calibration dataset in a comment.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import AudioQualityProfile, Stage2Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clipping detection
# ---------------------------------------------------------------------------

#: Standard forensic audio clipping threshold in dBFS.
CLIPPING_THRESHOLD_DBFS: float = -1.0

#: Linear amplitude equivalent: 10^(-1/20) ≈ 0.8913.
_CLIPPING_THRESHOLD_LINEAR: float = 10 ** (CLIPPING_THRESHOLD_DBFS / 20.0)


# ---------------------------------------------------------------------------
# Frame parameters
# ---------------------------------------------------------------------------

#: Frame duration in milliseconds. 20ms is standard for speech processing.
_FRAME_DURATION_MS: float = 20.0

#: Epsilon to prevent log(0) errors.
_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Energy-based VAD parameters
# ---------------------------------------------------------------------------

#: Frames at or below this RMS percentile are classified as silence.
_SILENCE_PERCENTILE: float = 20.0

#: Frames at or above this RMS percentile are classified as speech.
#: Frames between the two thresholds are ambiguous and excluded.
_SPEECH_PERCENTILE: float = 50.0

#: CV threshold above which the noise floor is considered unstable.
#: At CV > 0.30 the "silence" frames contain too much energy variation
#: to represent a stationary noise floor — the SNR estimate is unreliable.
_NOISE_INSTABILITY_THRESHOLD: float = 0.30


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


# ---------------------------------------------------------------------------
# Core streaming pass
# ---------------------------------------------------------------------------


def _stream_frame_rms(
    working_audio_path: Path,
    sample_rate: int,
) -> tuple[list[float], float] | None:
    """
    Stream the working WAV via soundfile.blocks() and compute per-frame
    RMS energy on the fly.

    This is the single I/O pass for Stage 2. Both clipping detection and
    SNR estimation derive from the data collected here.

    Memory profile
    --------------
    At any moment, one 20ms frame is held as a numpy array. At 48kHz
    stereo that is (960, 2) float32 ≈ 7.5 KB. The frame is processed
    immediately and discarded. What accumulates is the RMS list:
    one float32 per frame.

        2-hour file at 48kHz: ~360,000 frames → ~2.9 MB

    Stereo handling
    ---------------
    soundfile returns shape (n_samples, n_channels) for multi-channel
    audio. Multi-channel blocks are averaged across the channel axis
    to produce a mono representation before RMS computation:

        block.mean(axis=1)  →  (n_samples,)

    This per-frame downmix uses ~7.5 KB at a time — no full-file
    stereo-to-mono array is ever allocated.

    Parameters
    ----------
    working_audio_path : Path
        Path to the working PCM WAV produced by extraction.py.
    sample_rate : int
        Original sample rate from Stage 1 AudioProperties. Used to
        compute the correct frame size in samples.

    Returns
    -------
    tuple[list[float], float] | None
        (rms_per_frame, peak_linear_amplitude)
        None if soundfile fails to open or read the file.
    """
    # Frame size in samples at the original sample rate.
    frame_samples = int(sample_rate * _FRAME_DURATION_MS / 1000.0)

    rms_values: list[float] = []
    peak_linear: float = 0.0

    try:
        # soundfile.blocks() streams non-overlapping blocks of frame_samples.
        # overlap=0 is the default but stated explicitly for clarity.
        for block in sf.blocks(
            working_audio_path,
            blocksize=frame_samples,
            overlap=0,
            dtype="float32",
        ):
            if block.size == 0:
                continue

            # Downmix to mono on the fly if multi-channel.
            # block.ndim == 1 for mono, 2 for stereo/multichannel.
            if block.ndim == 2:
                block = block.mean(axis=1)

            frame_rms = float(np.sqrt(np.mean(block ** 2)))
            rms_values.append(frame_rms)

            frame_peak = float(np.max(np.abs(block)))
            if frame_peak > peak_linear:
                peak_linear = frame_peak

    except sf.SoundFileError as e:
        logger.warning(
            "soundfile failed to read %s: %s — "
            "clipping and SNR measurements unavailable.",
            working_audio_path.name, e,
        )
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error reading %s: %s — "
            "clipping and SNR measurements unavailable.",
            working_audio_path.name, e,
        )
        return None

    if not rms_values:
        logger.warning(
            "No audio frames read from %s. "
            "File may contain no audio content.",
            working_audio_path.name,
        )
        return None

    logger.debug(
        "Streamed %s: %d frames at %d Hz (%.1f seconds), peak: %.4f",
        working_audio_path.name,
        len(rms_values),
        sample_rate,
        len(rms_values) * _FRAME_DURATION_MS / 1000.0,
        peak_linear,
    )

    return rms_values, peak_linear


# ---------------------------------------------------------------------------
# Clipping detection
# ---------------------------------------------------------------------------


def _detect_clipping(peak_linear: float) -> tuple[bool, float]:
    """
    Determine whether peak amplitude exceeds the clipping threshold.

    Parameters
    ----------
    peak_linear : float
        Maximum absolute amplitude across all frames in [0.0, 1.0].

    Returns
    -------
    tuple[bool, float]
        (clipping_detected, peak_dbfs)
    """
    peak_dbfs = 20.0 * np.log10(peak_linear + _EPSILON)
    clipping = peak_dbfs >= CLIPPING_THRESHOLD_DBFS

    if clipping:
        logger.warning(
            "Clipping detected: peak %.2f dBFS (threshold: %.1f dBFS).",
            peak_dbfs, CLIPPING_THRESHOLD_DBFS,
        )
    else:
        logger.debug("No clipping. Peak: %.2f dBFS", peak_dbfs)

    return clipping, float(peak_dbfs)


# ---------------------------------------------------------------------------
# Noise floor stationarity check
# ---------------------------------------------------------------------------


def _check_noise_stability(silence_rms: np.ndarray) -> tuple[float | None, bool]:
    """
    Measure whether the detected noise floor is stationary using the
    Coefficient of Variation (CV).

    CV = std(silence_rms) / mean(silence_rms)

    CV measures whether the silence-frame RMS values are tightly clustered
    (flat, consistent noise floor) or widely spread (variable, chaotic).
    Unlike raw standard deviation, CV is dimensionless — it scales with
    the signal level, making it meaningful across files with very different
    loudness profiles.

    Why stationarity matters for SNR forensic validity
    ---------------------------------------------------
    SNR = 20 * log10(mean_speech_rms / mean_silence_rms) assumes the
    noise floor is stationary — that `mean_silence_rms` is a stable
    representative value of the background noise throughout the file.

    If the noise floor is non-stationary (e.g. rain, wind gusts, passing
    traffic, distant shouting), the p20 percentile captures the quietest
    moments between bursts rather than a true noise floor. The resulting
    SNR will be over-optimistic — the recording sounds better than it is.

    Example: a BWC recording in heavy rain.
        - p20 frames catch gaps between the loudest raindrops.
        - SNR might calculate to a "GOOD" 22 dB because the officer's
          voice is loud relative to those quiet gaps.
        - CV will be high because rain energy is highly variable.
        - noise_is_unstable=True warns downstream: don't trust the 22 dB.

    CV interpretation (thresholds are starting hypotheses — calibrate
    against a real BWC corpus):
        CV < 0.15   Stable, stationary noise floor. SNR is trustworthy.
        0.15–0.30   Moderate variation. SNR is reasonable but not precise.
        CV > 0.30   Unstable noise floor. SNR may be over-optimistic.

    This check is applied AFTER _MIN_SILENCE_FRAMES is verified.
    With fewer than ~10 silence frames, std() is statistically meaningless
    and CV would be noise. The caller is responsible for this guard.

    Parameters
    ----------
    silence_rms : np.ndarray
        RMS values of frames classified as silence by the VAD pass.
        Must have len >= _MIN_SILENCE_FRAMES (caller's responsibility).

    Returns
    -------
    tuple[float | None, bool]
        (noise_cv, is_unstable)
        noise_cv: the computed CV value. None if computation fails.
        is_unstable: True when noise_cv > _NOISE_INSTABILITY_THRESHOLD.
    """
    if len(silence_rms) == 0:
        return None, True

    mean_silence = float(np.mean(silence_rms))
    std_silence = float(np.std(silence_rms))

    # _EPSILON prevents division by zero for effectively-silent noise floors.
    noise_cv = std_silence / (mean_silence + _EPSILON)
    is_unstable = noise_cv > _NOISE_INSTABILITY_THRESHOLD

    if is_unstable:
        logger.warning(
            "Unstable noise floor detected: CV=%.3f (threshold: %.2f). "
            "The silence frames show high energy variation — the SNR "
            "estimate may be over-optimistic. Possible causes: rain, "
            "wind gusts, distant shouting, non-stationary background noise.",
            noise_cv,
            _NOISE_INSTABILITY_THRESHOLD,
        )
    else:
        logger.debug(
            "Noise floor is stable: CV=%.3f (threshold: %.2f).",
            noise_cv,
            _NOISE_INSTABILITY_THRESHOLD,
        )

    return float(noise_cv), is_unstable


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------


def _estimate_snr(
    rms_values: list[float],
) -> tuple[float | None, float | None, str | None, dict[str, float], float | None, bool]:
    """
    Estimate SNR from the per-frame RMS list using percentile-based VAD.

    The RMS list is treated as an energy histogram. Speech and silence
    frames are identified by their position in the distribution rather
    than a fixed threshold, making the classification adaptive to each
    file's own energy profile.

        silence:   frames at or below _SILENCE_PERCENTILE (20th pct)
        speech:    frames at or above _SPEECH_PERCENTILE  (50th pct)
        ambiguous: frames in between — excluded from SNR computation

    SNR = 20 * log10(mean_speech_rms / mean_silence_rms)

    After computing SNR, a stationarity check is run on the silence
    frames to validate whether the noise floor is reliable. See
    _check_noise_stability() for full rationale.

    Parameters
    ----------
    rms_values : list[float]
        Per-frame RMS values from _stream_frame_rms().

    Returns
    -------
    tuple[float|None, float|None, str|None, dict, float|None, bool]
        (snr_db, speech_fraction, note, energy_percentiles,
         noise_cv, noise_is_unstable)
        snr_db: SNR in dB, or None if estimation failed.
        speech_fraction: fraction of frames classified as speech.
        note: human-readable caveat, or None if clean.
        energy_percentiles: {p10, p20, p50, p80, p90} for output JSON.
        noise_cv: Coefficient of Variation of silence frames, or None.
        noise_is_unstable: True when noise_cv > _NOISE_INSTABILITY_THRESHOLD.
    """
    rms_array = np.array(rms_values, dtype=np.float32)
    total_frames = len(rms_array)

    # Compute energy percentile summary for output metadata.
    pct_keys = [10, 20, 50, 80, 90]
    energy_percentiles = {
        f"p{p}": float(np.percentile(rms_array, p))
        for p in pct_keys
    }

    # Default stability values — overwritten below if computation succeeds.
    noise_cv: float | None = None
    noise_is_unstable: bool = False

    if total_frames < 10:
        note = (
            f"Only {total_frames} frame(s) — "
            "audio too short for reliable SNR estimation."
        )
        logger.warning(note)
        return None, None, note, energy_percentiles, noise_cv, noise_is_unstable

    silence_threshold = np.percentile(rms_array, _SILENCE_PERCENTILE)
    speech_threshold = np.percentile(rms_array, _SPEECH_PERCENTILE)

    silence_mask = rms_array <= silence_threshold
    speech_mask = rms_array >= speech_threshold

    silence_rms = rms_array[silence_mask]
    speech_rms = rms_array[speech_mask]
    speech_fraction = float(np.sum(speech_mask)) / total_frames

    logger.debug(
        "VAD: %d total | %d speech | %d silence | %d ambiguous | "
        "speech fraction: %.2f",
        total_frames, len(speech_rms), len(silence_rms),
        total_frames - len(speech_rms) - len(silence_rms),
        speech_fraction,
    )

    if len(silence_rms) < _MIN_SILENCE_FRAMES:
        note = (
            f"Only {len(silence_rms)} silence frame(s) found "
            f"(minimum: {_MIN_SILENCE_FRAMES}). "
            "Audio may be continuous speech — SNR estimate unreliable."
        )
        logger.warning("SNR skipped: %s", note)
        return None, speech_fraction, note, energy_percentiles, noise_cv, noise_is_unstable

    if len(speech_rms) == 0:
        note = (
            "No speech frames detected above the energy threshold. "
            "Audio may be near-silent or contain only background noise."
        )
        logger.warning("SNR skipped: %s", note)
        return None, speech_fraction, note, energy_percentiles, noise_cv, noise_is_unstable

    # ------------------------------------------------------------------
    # Noise floor stationarity check.
    # Applied after _MIN_SILENCE_FRAMES guard — std() is statistically
    # meaningless on very small samples.
    # ------------------------------------------------------------------
    noise_cv, noise_is_unstable = _check_noise_stability(silence_rms)

    mean_speech_rms = float(np.mean(speech_rms))
    mean_silence_rms = float(np.mean(silence_rms))

    if mean_silence_rms < _EPSILON:
        note = (
            "Noise floor is effectively zero — audio may have been "
            "pre-processed. SNR capped at 60 dB."
        )
        logger.debug(note)
        return 60.0, speech_fraction, note, energy_percentiles, noise_cv, noise_is_unstable

    snr_db = 20.0 * np.log10(mean_speech_rms / mean_silence_rms)

    # If noise is unstable, append a caveat to the note without
    # changing the SNR value itself. The raw measurement is preserved.
    # Phase 4 Fusion uses noise_is_unstable independently.
    note = None
    if noise_is_unstable:
        note = (
            f"Noise floor is non-stationary (CV={noise_cv:.3f} > "
            f"{_NOISE_INSTABILITY_THRESHOLD}). SNR of {snr_db:.1f} dB "
            "may be over-optimistic — background environment is chaotic. "
            "Phase 4 Fusion should weight this file's confidence scores down."
        )

    logger.debug(
        "SNR: %.1f dB (speech RMS: %.6f, noise RMS: %.6f, noise CV: %s)",
        snr_db, mean_speech_rms, mean_silence_rms,
        f"{noise_cv:.3f}" if noise_cv is not None else "n/a",
    )

    return float(snr_db), speech_fraction, note, energy_percentiles, noise_cv, noise_is_unstable


# ---------------------------------------------------------------------------
# SNR classification
# ---------------------------------------------------------------------------


def _classify_snr(snr_db: float | None) -> tuple[str, bool]:
    """
    Map SNR to classification label and flag status.

    Parameters
    ----------
    snr_db : float | None
        Estimated SNR in dB. None when SNR could not be computed.

    Returns
    -------
    tuple[str, bool]
        (classification_label, snr_flagged)
    """
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

    Single soundfile.blocks() streaming pass through the working audio.
    Clipping detection and SNR estimation both derive from the same
    frame-by-frame read — the file is never read twice.

    Parameters
    ----------
    stage1_result : Stage1Result
        Complete output from run_stage1(). Provides original audio
        properties (sample rate, channels, duration) without re-probing.
    working_audio_path : Path
        Path to the working PCM WAV produced by extract_working_audio().
        May be the original file (for native soundfile formats) or the
        extracted WAV (for video containers and compressed audio).

    Returns
    -------
    Stage2Result
        Complete Stage 2 profiling results. Pass to run_stage3().

    Examples
    --------
    >>> from src.utils.audio import run_stage1, run_stage2
    >>> from src.utils.audio.extraction import extract_working_audio
    >>>
    >>> s1 = run_stage1(Path("footage/incident_001.mp4"))
    >>> working_path, _, _ = extract_working_audio(
    ...     s1.original_path, Path("data"), "incident_001"
    ... )
    >>> s2 = run_stage2(s1, working_path)
    >>> s2.audio_quality.snr_db
    18.4
    >>> s2.audio_quality.snr_classification
    'MODERATE'
    """
    from .stage1 import Stage1Result  # Avoid circular import.

    path = stage1_result.original_path
    props = stage1_result.audio_properties

    logger.info("Stage 2 — Audio Quality Profiling: %s", path.name)
    logger.debug("Working audio: %s", working_audio_path)

    # ------------------------------------------------------------------
    # Step 1: Source basic properties from Stage 1 — no re-probe needed.
    # ------------------------------------------------------------------
    duration_seconds = props.duration_seconds
    sample_rate_original = props.sample_rate
    channels_original = props.channels

    # ------------------------------------------------------------------
    # Step 2: Single soundfile.blocks() streaming pass.
    # Collects per-frame RMS values and peak amplitude in one read.
    # ------------------------------------------------------------------
    stream_result = _stream_frame_rms(working_audio_path, sample_rate_original)

    if stream_result is None:
        logger.warning(
            "Streaming pass failed for %s — "
            "clipping and SNR measurements unavailable.",
            path.name,
        )
        clipping_detected = False
        peak_dbfs = None
        snr_db = None
        speech_fraction = None
        snr_note = "Waveform streaming failed — measurements unavailable."
        energy_percentiles: dict[str, float] = {}
        noise_cv: float | None = None
        noise_is_unstable: bool = False
    else:
        rms_values, peak_linear = stream_result

        # ------------------------------------------------------------------
        # Step 3: Clipping detection from peak amplitude.
        # ------------------------------------------------------------------
        clipping_detected, peak_dbfs = _detect_clipping(peak_linear)

        # ------------------------------------------------------------------
        # Step 4: SNR estimation from RMS histogram.
        # ------------------------------------------------------------------
        snr_db, speech_fraction, snr_note, energy_percentiles, noise_cv, noise_is_unstable = (
            _estimate_snr(rms_values)
        )

    # ------------------------------------------------------------------
    # Step 5: Classify SNR.
    # ------------------------------------------------------------------
    snr_classification, snr_flagged = _classify_snr(snr_db)

    if snr_flagged and snr_db is not None:
        logger.warning(
            "Low SNR: %.1f dB (%s) — diarization reliability may be "
            "reduced for %s.",
            snr_db, snr_classification, path.name,
        )

    # ------------------------------------------------------------------
    # Step 6: Assemble result.
    # ------------------------------------------------------------------
    audio_quality = AudioQualityProfile(
        duration_seconds=duration_seconds,
        sample_rate_original=sample_rate_original,
        channels_original=channels_original,
        clipping_detected=clipping_detected,
        clipping_peak_dbfs=peak_dbfs,
        snr_db=snr_db,
        snr_classification=snr_classification,
        snr_flagged=snr_flagged,
        vad_speech_fraction=speech_fraction,
        vad_energy_percentiles=energy_percentiles,
        noise_stability_cv=noise_cv,
        noise_is_unstable=noise_is_unstable,
        snr_note=snr_note,
    )

    result = Stage2Result(audio_quality=audio_quality)

    logger.info(
        "Stage 2 complete — SNR: %s (%s) | clipping: %s | speech: %s",
        f"{snr_db:.1f} dB" if snr_db is not None else "unknown",
        snr_classification,
        clipping_detected,
        f"{speech_fraction:.0%}" if speech_fraction is not None else "unknown",
    )

    return result