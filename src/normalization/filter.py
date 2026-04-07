"""
src/utils/normalization/filter.py
===================================
TrustScript Phase 1 — Stage 3 ffmpeg Filter Chain

Responsibility
--------------
Build and execute the ffmpeg normalization filter chain.
This is the only module in TrustScript that writes a new audio file
derived from the original signal. Everything before this stage is
read-only.

Filter chain (applied in order)
--------------------------------
1. Extract audio stream   -vn discards video track
2. Mono downmix           pan=mono|c0=0.5*c0+0.5*c1  (stereo only)
3. Resample               aresample=16000
4. EBU R128 normalization loudnorm two-pass (-16 LUFS)
5. Output                 pcm_s16le WAV

Why two-pass loudnorm?
-----------------------
A single-pass loudnorm measures and adjusts in the same pass using an
estimate of the integrated loudness. This can overshoot or undershoot
the target on files with unusual loudness distributions (very dynamic
BWC audio, long silences, clipped sections).

Two-pass loudnorm:
    Pass 1 (analysis): measure the true integrated loudness, LRA,
                       and true peak of the entire file.
    Pass 2 (apply):    apply the exact gain correction needed to hit
                       the target, using Pass 1 measurements.

The result is a more accurate final loudness, especially on BWC footage
where loudness may vary dramatically between quiet patrol and active
incident sections.

Intentionally excluded
-----------------------
    - No noise gate
    - No FFT denoising (afftdn)
    - No spectral subtraction

Denoising is deferred until post-Phase 1. The baseline must be
established before introducing corrective processing.

Contributors
------------
    All subprocess calls to ffmpeg live here. Do not spawn ffmpeg
    processes from any other module. If you add a new filter, add
    its corresponding bias label to constants.py.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    FFMPEG_TIMEOUT_SECONDS,
    LOUDNORM_TARGET_I,
    LOUDNORM_TARGET_LRA,
    LOUDNORM_TARGET_TP,
    OUTPUT_CODEC,
    TARGET_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass 1 result
# ---------------------------------------------------------------------------


@dataclass
class LoudnormMeasurement:
    """
    Loudness measurements from the ffmpeg loudnorm analysis pass.

    These values are fed directly into Pass 2 as input parameters,
    ensuring the correction is computed from the actual file content
    rather than an estimate.

    Attributes
    ----------
    input_i : float
        Measured integrated loudness in LUFS.
    input_lra : float
        Measured Loudness Range in LU.
    input_tp : float
        Measured True Peak in dBTP.
    input_thresh : float
        Loudness threshold used during measurement.
    target_offset : float
        Gain offset (dB) required to reach the target loudness.
    """
    input_i: float
    input_lra: float
    input_tp: float
    input_thresh: float
    target_offset: float


# ---------------------------------------------------------------------------
# Pass 1 — loudness analysis
# ---------------------------------------------------------------------------


def measure_loudness(source: Path) -> LoudnormMeasurement:
    """
    Run ffmpeg loudnorm in analysis mode to measure integrated loudness.

    This is Pass 1 of the two-pass normalization. It decodes the audio
    and measures loudness without writing any output file. The result
    is used as input parameters for Pass 2.

    Parameters
    ----------
    source : Path
        Path to the working audio file (original rate, original channels).
        May be the original file for passthrough-native formats or the
        extracted working WAV.

    Returns
    -------
    LoudnormMeasurement
        Measured loudness values from the analysis pass.

    Raises
    ------
    RuntimeError
        If ffmpeg fails or loudnorm JSON cannot be parsed from output.
    """
    cmd = [
        "ffmpeg",
        "-i", str(source),
        "-vn",
        "-af", (
            f"loudnorm="
            f"I={LOUDNORM_TARGET_I}:"
            f"TP={LOUDNORM_TARGET_TP}:"
            f"LRA={LOUDNORM_TARGET_LRA}:"
            f"print_format=json"
        ),
        "-f", "null",
        "-loglevel", "info",   # loudnorm JSON is written to stderr at info level
        "-",
    ]

    logger.debug("Pass 1 (loudness analysis): %s", source.name)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg loudness analysis timed out after "
            f"{FFMPEG_TIMEOUT_SECONDS}s for: {source.name}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on your PATH. "
            "See README.md for installation instructions."
        )

    # loudnorm writes JSON to stderr. Extract the JSON block.
    stderr = result.stderr
    json_start = stderr.rfind("{")
    json_end = stderr.rfind("}") + 1

    if json_start == -1 or json_end == 0:
        raise RuntimeError(
            f"Could not find loudnorm JSON in ffmpeg output for: {source.name}\n"
            f"ffmpeg stderr (last 500 chars):\n{stderr[-500:]}"
        )

    try:
        data = json.loads(stderr[json_start:json_end])
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse loudnorm JSON for: {source.name}\n"
            f"JSON candidate: {stderr[json_start:json_end][:200]}"
        ) from e

    measurement = LoudnormMeasurement(
        input_i=float(data.get("input_i", -70.0)),
        input_lra=float(data.get("input_lra", 0.0)),
        input_tp=float(data.get("input_tp", -70.0)),
        input_thresh=float(data.get("input_thresh", -70.0)),
        target_offset=float(data.get("target_offset", 0.0)),
    )

    logger.debug(
        "Pass 1 complete: %.1f LUFS, LRA %.1f LU, TP %.1f dBTP",
        measurement.input_i,
        measurement.input_lra,
        measurement.input_tp,
    )

    return measurement


# ---------------------------------------------------------------------------
# Pass 2 — normalization
# ---------------------------------------------------------------------------


def apply_normalization(
    source: Path,
    dest: Path,
    measurement: LoudnormMeasurement,
    source_channels: int,
) -> None:
    """
    Run ffmpeg to apply the full normalization filter chain.

    This is Pass 2. It uses the measurements from Pass 1 to apply the
    exact gain correction needed, then resamples and downmixes.

    Filter chain applied (in order):
        1. -vn          Discard video stream
        2. pan          Mono downmix (stereo sources only)
        3. aresample    Downsample to 16kHz
        4. loudnorm     Apply measured correction (linear mode)

    The loudnorm filter is run in linear mode in Pass 2, using the
    exact measurements from Pass 1. This is more accurate than the
    default dynamic mode for files with non-standard loudness profiles.

    Parameters
    ----------
    source : Path
        Working audio file to normalise.
    dest : Path
        Output path for the normalised WAV.
    measurement : LoudnormMeasurement
        Pass 1 measurements used to compute exact gain correction.
    source_channels : int
        Number of channels in the source file. If > 1, mono downmix
        filter is prepended to the chain.

    Raises
    ------
    RuntimeError
        If ffmpeg returns a non-zero exit code or times out.
    """
    # Build the audio filter chain as a list of filter strings.
    # Joining with commas applies them in sequence.
    filters: list[str] = []

    # Step 1: Mono downmix (only when source is not already mono).
    if source_channels > 1:
        filters.append("pan=mono|c0=0.5*c0+0.5*c1")

    # Step 2: Resample to 16kHz.
    filters.append(f"aresample={TARGET_SAMPLE_RATE}")

    # Step 3: Apply loudnorm using Pass 1 measurements (linear mode).
    # measured_* parameters lock in the Pass 1 values so loudnorm
    # applies the exact correction rather than re-measuring.
    filters.append(
        f"loudnorm="
        f"I={LOUDNORM_TARGET_I}:"
        f"TP={LOUDNORM_TARGET_TP}:"
        f"LRA={LOUDNORM_TARGET_LRA}:"
        f"measured_I={measurement.input_i}:"
        f"measured_LRA={measurement.input_lra}:"
        f"measured_TP={measurement.input_tp}:"
        f"measured_thresh={measurement.input_thresh}:"
        f"offset={measurement.target_offset}:"
        f"linear=true:"
        f"print_format=none"
    )

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg",
        "-i", str(source),
        "-vn",
        "-af", filter_str,
        "-acodec", OUTPUT_CODEC,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-y",                       # Overwrite output if it exists.
        "-loglevel", "error",
        str(dest),
    ]

    logger.debug(
        "Pass 2 (apply normalization): %s → %s",
        source.name, dest.name,
    )
    logger.debug("Filter chain: %s", filter_str)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg normalization timed out after "
            f"{FFMPEG_TIMEOUT_SECONDS}s for: {source.name}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on your PATH. "
            "See README.md for installation instructions."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg normalization failed for: {source.name}\n"
            f"Return code: {result.returncode}\n"
            f"stderr: {result.stderr.strip()}"
        )

    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg produced no output file: {dest}\n"
            "The source may contain no audio or the filter chain failed silently."
        )

    logger.debug(
        "Pass 2 complete: %s (%.1f MB)",
        dest.name,
        dest.stat().st_size / 1_048_576,
    )