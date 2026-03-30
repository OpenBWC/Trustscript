"""
src/utils/audio/probe.py
=========================
TrustScript Phase 1 — Audio Property Detection via ffprobe/ffmpeg

Responsibility
--------------
All subprocess interactions with ffprobe and ffmpeg that inspect
(but never modify) audio files. This module is the only place in
TrustScript that calls ffprobe or ffmpeg directly.

Two operations are implemented:

    probe_audio_properties()
        Fast, metadata-only inspection via ffprobe.
        Does not decode the audio stream. Returns sample rate,
        channel count, codec, and duration.

    measure_loudness()
        Full-decode loudness analysis via ffmpeg's loudnorm filter.
        Required to make the passthrough decision when the cheaper
        property checks (sample rate, channels, codec) already pass.
        Slower than probing — only called when necessary.

System dependency
-----------------
    ffmpeg and ffprobe must be installed at the OS level.
    These are not pip-installable. See README.md for instructions.

Contributors
------------
    Keep subprocess calls contained within this module. If you need
    a new ffprobe or ffmpeg inspection feature, add it here rather
    than spawning subprocesses from other modules. Do not add
    normalization or audio-writing logic here — that belongs in
    Stage 3 (normalization.py, to be added).
"""

import json
import logging
import subprocess
from pathlib import Path

from .exceptions import AudioIngestionError, FFprobeError
from .models import AudioProperties, LoudnessProperties

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_ffprobe() -> None:
    """
    Verify that ffprobe is available on the system PATH.

    Called at the start of any function that needs ffprobe. Fails fast
    with a clear installation message rather than letting the subprocess
    call produce a cryptic FileNotFoundError.

    Raises
    ------
    FFprobeError
        If ffprobe cannot be found on the system PATH.
    """
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise FFprobeError(
            "ffprobe is not installed or not on your PATH.\n"
            "TrustScript requires ffmpeg/ffprobe at the OS level.\n"
            "  macOS:          brew install ffmpeg\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  Windows:        choco install ffmpeg\n"
            "See README.md for full installation instructions."
        )


def _require_ffmpeg() -> None:
    """
    Verify that ffmpeg is available on the system PATH.

    Raises
    ------
    FFprobeError
        If ffmpeg cannot be found on the system PATH.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise FFprobeError(
            "ffmpeg is not installed or not on your PATH.\n"
            "TrustScript requires ffmpeg at the OS level.\n"
            "  macOS:          brew install ffmpeg\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  Windows:        choco install ffmpeg\n"
            "See README.md for full installation instructions."
        )


# ---------------------------------------------------------------------------
# Public probe functions
# ---------------------------------------------------------------------------


def probe_audio_properties(path: Path) -> AudioProperties:
    """
    Extract audio stream properties from a file using ffprobe.

    This is a read-only metadata inspection pass. ffprobe reads stream
    headers without decoding the audio content, making it fast and
    non-destructive regardless of file size.

    For video containers (.mp4, .mov, .avi, .mkv), ffprobe inspects all
    streams and returns the properties of the first audio stream found.
    The video stream is not read or decoded.

    ffprobe is invoked with:
        -v quiet            Suppress all log output except errors.
        -print_format json  Return structured JSON output.
        -show_streams       Include per-stream metadata.

    Parameters
    ----------
    path : Path
        Path to the input file. Must exist and be readable.

    Returns
    -------
    AudioProperties
        Structured audio properties from the first audio stream.

    Raises
    ------
    FFprobeError
        If ffprobe is not installed, the file is unreadable, ffprobe
        returns a non-zero exit code, the output is not valid JSON,
        or no audio stream is found in the file.

    Examples
    --------
    >>> props = probe_audio_properties(Path("footage/incident_001.mp4"))
    >>> props.sample_rate
    48000
    >>> props.channels
    2
    >>> props.codec_name
    'aac'
    """
    _require_ffprobe()

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise FFprobeError(
            f"ffprobe failed on: {path.name}\n"
            f"Return code: {e.returncode}\n"
            f"stderr: {e.stderr.strip()}\n"
            "The file may be corrupt, unreadable, or in an unexpected format."
        ) from e

    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise FFprobeError(
            f"ffprobe returned invalid JSON for: {path.name}\n"
            f"Raw output (first 200 chars): {result.stdout[:200]}"
        ) from e

    # Find the first audio stream. Video and subtitle streams are skipped.
    audio_stream = next(
        (s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )

    if audio_stream is None:
        raise FFprobeError(
            f"No audio stream found in: {path.name}\n"
            "Verify the file contains audio and is not video-only or corrupt."
        )

    # Duration may be on the stream or at the container (format) level.
    # Some streaming formats cannot provide duration without full decode.
    duration = float(
        audio_stream.get("duration")
        or probe_data.get("format", {}).get("duration", 0.0)
    )

    # Bit rate is not available for all formats (e.g. lossless codecs).
    raw_bit_rate = audio_stream.get("bit_rate")
    bit_rate = int(raw_bit_rate) if raw_bit_rate is not None else None

    properties = AudioProperties(
        sample_rate=int(audio_stream.get("sample_rate", 0)),
        channels=int(audio_stream.get("channels", 0)),
        codec_name=audio_stream.get("codec_name", "unknown"),
        duration_seconds=duration,
        bit_rate=bit_rate,
    )

    logger.debug(
        "Probed %s: %dHz, %dch, codec=%s, duration=%.1fs",
        path.name,
        properties.sample_rate,
        properties.channels,
        properties.codec_name,
        properties.duration_seconds,
    )

    return properties


def measure_loudness(path: Path) -> LoudnessProperties:
    """
    Measure integrated loudness using ffmpeg's loudnorm filter in
    analysis mode.

    Unlike probe_audio_properties(), this requires a full decode pass
    through the audio content. It is slower in proportion to file length.
    For a 60-minute BWC file this may take 30–90 seconds on CPU.

    This function does not write any output. The -f null - flag
    discards all decoded audio. Only the loudnorm JSON measurements
    written to stderr are captured.

    When to call this
    -----------------
    Only call this function when the cheaper property checks (sample rate,
    channels, codec) have already passed and loudness is the only remaining
    question for the passthrough decision. See stage1.py for the
    conditional logic that gates this call.

    loudnorm JSON output format (written to stderr by ffmpeg):
        {
          "input_i":   "-23.45",   Integrated loudness (LUFS)
          "input_lra":  "7.30",   Loudness Range (LU)
          "input_tp":  "-2.10",   True Peak (dBTP)
          "input_thresh": "...",  Internal threshold values
          ...
        }

    Parameters
    ----------
    path : Path
        Path to the audio or video file to measure.

    Returns
    -------
    LoudnessProperties
        Measured loudness values from the loudnorm analysis pass.

    Raises
    ------
    FFprobeError
        If ffmpeg is not installed, fails to run, or the loudnorm JSON
        cannot be found or parsed in stderr output.

    Examples
    --------
    >>> loudness = measure_loudness(Path("footage/incident_001.wav"))
    >>> loudness.integrated_lufs
    -23.4
    >>> loudness.true_peak
    -2.1
    """
    _require_ffmpeg()

    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-af", "loudnorm=print_format=json",
        "-f", "null",
        "-",
    ]

    # ffmpeg exits with a non-zero code when writing to /dev/null ("-"),
    # so we do not use check=True here. We inspect stderr content instead.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # loudnorm writes its JSON block to stderr, surrounded by ffmpeg's
    # standard log output. Locate the JSON block by finding the outermost
    # braces. Using rfind() targets the last occurrence, which is the
    # loudnorm JSON (ffmpeg may emit earlier JSON-like strings in headers).
    stderr = result.stderr
    json_start = stderr.rfind("{")
    json_end = stderr.rfind("}") + 1

    if json_start == -1 or json_end == 0:
        raise FFprobeError(
            f"Could not find loudnorm JSON in ffmpeg output for: {path.name}\n"
            f"ffmpeg stderr (last 500 chars):\n{stderr[-500:]}"
        )

    try:
        loudnorm_data = json.loads(stderr[json_start:json_end])
    except json.JSONDecodeError as e:
        raise FFprobeError(
            f"Failed to parse loudnorm JSON for: {path.name}\n"
            f"JSON candidate: {stderr[json_start:json_end][:200]}"
        ) from e

    loudness = LoudnessProperties(
        integrated_lufs=float(loudnorm_data.get("input_i", 0.0)),
        loudness_range=float(loudnorm_data.get("input_lra", 0.0)),
        true_peak=float(loudnorm_data.get("input_tp", 0.0)),
    )

    logger.debug(
        "Loudness for %s: %.1f LUFS, LRA %.1f LU, TP %.1f dBTP",
        path.name,
        loudness.integrated_lufs,
        loudness.loudness_range,
        loudness.true_peak,
    )

    return loudness