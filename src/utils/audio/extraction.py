"""
src/utils/audio/extraction.py
==============================
TrustScript Phase 1 — Audio Extraction (between Stage 1 and Stage 2)

Responsibility
--------------
Extract the audio stream from video containers and compressed audio
formats to a working PCM WAV file at the original sample rate and
channel count.

This is a format conversion step, not signal processing. The extracted
WAV is acoustically identical to the audio in the source file — same
sample rate, same channels, same loudness, same noise floor. Nothing
about the audio signal is changed.

Why this step exists
---------------------
soundfile — used for memory-efficient streaming in Stage 2 and
normalization in Stage 3 — cannot read video containers (.mp4, .mov,
.avi, .mkv) or compressed audio formats (.mp3, .m4a, .aac).

By extracting to a working WAV once, every subsequent stage can use
soundfile directly instead of spawning ffmpeg subprocesses per stage.
The working file is also reused across Stage 2 and Stage 3, avoiding
repeated reads of the original video container.

Formats that require extraction
---------------------------------
    Video containers: .mp4, .mov, .avi, .mkv  (audio stream extracted)
    Compressed audio: .mp3, .m4a              (decoded to PCM)

Formats used directly (no extraction)
--------------------------------------
    .wav   soundfile reads natively
    .flac  soundfile reads natively

Working file
------------
    Location:  {output_dir}/working/{incident_id}_raw.wav
    Format:    32-bit float PCM (pcm_f32le)
    Sample rate: ORIGINAL (not resampled — Stage 3 handles resampling)
    Channels:    ORIGINAL (not downmixed — Stage 3 handles downmix)

    pcm_f32le is used instead of pcm_s16le for two reasons:
        1. Precision: 24-bit or 32-bit source material loses no precision.
           16-bit quantization would introduce noise into the noise floor
           that Stage 2 is trying to measure.
        2. Clipping accuracy: pcm_f32le preserves samples that exceed
           0 dBFS at their true float values. pcm_s16le hard-clips them
           to ±32767 at conversion, masking the true peak level before
           Stage 2 can measure it.
    Storage cost is 2x vs 16-bit, which is acceptable for a temporary
    working artifact in a forensic pipeline.

Chain of custody
----------------
    The working WAV is a derived artifact. Its own SHA256 is recorded
    in the pipeline metadata alongside the original file's SHA256 from
    Stage 1, so the derivation chain is fully traceable.

Contributors
------------
    If a new input format is added to ALL_SUPPORTED_FORMATS in stage1.py,
    check whether it belongs in FORMATS_NEEDING_EXTRACTION below.
    soundfile supports: WAV, FLAC, OGG, AIFF, AU, RAW, MAT, W64, SD2,
    IRCAM, PVF, HTK, SDS, AVR, and some others.
    When in doubt, add it to FORMATS_NEEDING_EXTRACTION — ffmpeg handles
    all of them correctly.
"""

import hashlib
import logging
import subprocess
from pathlib import Path

from .exceptions import AudioIngestionError, FFprobeError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format classification
# ---------------------------------------------------------------------------

#: Formats that require ffmpeg extraction before soundfile can read them.
#: Video containers and compressed audio codecs both fall here.
FORMATS_NEEDING_EXTRACTION: frozenset[str] = frozenset({
    # Video containers — audio stream must be demuxed.
    ".mp4", ".mov", ".avi", ".mkv",
    # Compressed audio — must be decoded to PCM.
    ".mp3", ".m4a",
})

#: Formats soundfile can read natively. No extraction needed.
#: The original file is used directly as the working audio path.
FORMATS_SOUNDFILE_NATIVE: frozenset[str] = frozenset({
    ".wav", ".flac",
})


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------


def extract_working_audio(
    original_path: Path,
    output_dir: Path,
    incident_id: str,
) -> tuple[Path, bool, str]:
    """
    Produce a working PCM WAV file from the original input.

    For formats soundfile can read natively (.wav, .flac), the original
    file is returned directly — no file is written and no ffmpeg process
    is spawned. The `extracted` flag in the return value distinguishes
    these two cases so the caller can log and track provenance correctly.

    For all other formats (video containers, compressed audio), ffmpeg
    extracts or decodes the audio stream to a PCM WAV at the original
    sample rate and channel count. The working file is written to
    {output_dir}/working/{incident_id}_raw.wav.

    Parameters
    ----------
    original_path : Path
        Resolved absolute path to the original input file.
    output_dir : Path
        Root output directory for this pipeline run. The working
        subdirectory is created inside it automatically.
    incident_id : str
        Incident identifier used to name the working file.

    Returns
    -------
    tuple[Path, bool, str]
        (working_audio_path, extracted, working_sha256)
        working_audio_path: path to use for all subsequent stage reads.
        extracted: True if a new file was written, False if the original
                   is returned directly (soundfile-native format).
        working_sha256: SHA256 of the working audio file. For native
                        formats this matches Stage 1's original_sha256.
                        For extracted files it is the SHA256 of the new
                        WAV, documenting the derived artifact.

    Raises
    ------
    AudioIngestionError
        If ffmpeg fails to extract the audio stream.
    FFprobeError
        If ffmpeg is not installed at the OS level.

    Examples
    --------
    >>> path, extracted, sha256 = extract_working_audio(
    ...     Path("footage/incident_001.mp4"),
    ...     Path("data"),
    ...     "incident_001",
    ... )
    >>> extracted
    True
    >>> path
    PosixPath('data/working/incident_001_raw.wav')
    """
    suffix = original_path.suffix.lower()

    # ------------------------------------------------------------------
    # Fast path: soundfile-native format — no extraction needed.
    # ------------------------------------------------------------------
    if suffix in FORMATS_SOUNDFILE_NATIVE:
        logger.debug(
            "%s is soundfile-native (%s) — using original file directly.",
            original_path.name, suffix,
        )
        sha256 = _sha256(original_path)
        return original_path, False, sha256

    # ------------------------------------------------------------------
    # Extraction path: video container or compressed audio.
    # ------------------------------------------------------------------
    working_dir = output_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    working_path = working_dir / f"{incident_id}_raw.wav"

    logger.info(
        "Extracting audio from %s → %s",
        original_path.name, working_path.name,
    )

    _run_extraction(original_path, working_path)

    # Compute SHA256 of the working file for provenance documentation.
    working_sha256 = _sha256(working_path)

    logger.info(
        "Extraction complete: %s | SHA256: %s...",
        working_path.name, working_sha256[:12],
    )

    return working_path, True, working_sha256


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_extraction(source: Path, dest: Path) -> None:
    """
    Run ffmpeg to extract or decode audio to a PCM WAV file.

    The output preserves the original sample rate and channel count.
    No resampling, no downmixing, no loudness adjustment — those
    operations belong to Stage 3 (normalization).

    pcm_f32le (32-bit float PCM) is used as the output codec:
    - No quantization noise on high-bit-depth sources (24-bit FLAC,
      32-bit float recordings).
    - Samples exceeding 0 dBFS are preserved at their true float values
      rather than hard-clipped, giving Stage 2 accurate peak measurements.
    - soundfile.blocks(dtype='float32') reads pcm_f32le as a direct
      memory copy — no conversion overhead at read time.

    Parameters
    ----------
    source : Path
        Original input file (video container or compressed audio).
    dest : Path
        Destination path for the extracted WAV file.

    Raises
    ------
    FFprobeError
        If ffmpeg is not installed.
    AudioIngestionError
        If ffmpeg returns a non-zero exit code.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",               # Prevents ffmpeg from hanging on user prompts
        "-i", str(source),
        "-vn",                    # Discard video stream.
        "-acodec", "pcm_f32le",   # 32-bit float PCM — preserves full
                                  # source precision including samples
                                  # that exceed 0 dBFS. Unlike pcm_s16le,
                                  # no quantization noise is introduced
                                  # and no hard-clipping occurs at the
                                  # conversion step. Critical for accurate
                                  # Stage 2 noise floor and clipping
                                  # measurements on high-bit-depth sources
                                  # (24-bit FLAC, 32-bit float recordings).
                                  # File size is 2x vs 16-bit but this
                                  # is a temporary working artifact.
        "-y",                     # Overwrite destination if it exists
                                  # (resuming a failed run).
        "-loglevel", "error",     # Suppress normal ffmpeg output.
        str(dest),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,          # 10-minute timeout for long files.
        )
    except FileNotFoundError:
        raise FFprobeError(
            "ffmpeg is not installed or not on your PATH. "
            "TrustScript requires ffmpeg at the OS level. "
            "See README.md for installation instructions."
        )
    except subprocess.TimeoutExpired:
        raise AudioIngestionError(
            f"ffmpeg timed out extracting audio from: {source.name}\n"
            "The file may be unusually large or the system is under load."
        )

    if result.returncode != 0:
        raise AudioIngestionError(
            f"ffmpeg failed to extract audio from: {source.name}\n"
            f"Return code: {result.returncode}\n"
            f"stderr: {result.stderr.strip()}"
        )

    if not dest.exists() or dest.stat().st_size == 0:
        raise AudioIngestionError(
            f"ffmpeg completed but produced no output file: {dest}\n"
            "The source file may contain no audio stream."
        )

def _sha256(path: Path) -> str:
    """
    Compute SHA256 using a reusable buffer to minimize memory allocations.
    Optimal for the large f32le WAV files generated in this stage.
    """
    hasher = hashlib.sha256()
    chunk_size = 128 * 1024  # 128KB matches modern SSD block sizes
    
    # buffering=0 is faster for raw binary reads on large files
    with open(path, "rb", buffering=0) as f:
        # Pre-allocate one single buffer in RAM
        buffer = bytearray(chunk_size)
        mv = memoryview(buffer)
        
        while n := f.readinto(buffer):
            # Update the hasher using the same memory space
            hasher.update(mv[:n])
            
    return hasher.hexdigest()