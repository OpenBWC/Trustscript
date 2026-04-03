"""
src/utils/audio/stage1.py
==========================
TrustScript Phase 1 — Stage 1 Orchestrator: Input Handling & Format Detection

Responsibility
--------------
Orchestrates the complete Stage 1 execution: validates the input,
hashes it, probes its properties, conditionally measures loudness,
makes the passthrough decision, and returns a structured result.

This module also owns:
    - Format constants (supported extensions, normalization targets)
    - Dataclasses for structured Stage 1 output
    - The passthrough decision logic (check_passthrough)
    - The public run_stage1() entry point

It does not perform hashing or subprocess calls directly — those are
delegated to hashing.py and probe.py respectively.

Stage 1 is strictly read-only. No files are created or modified.

Contributors
------------
    - Add new supported formats to SUPPORTED_VIDEO_FORMATS or
      SUPPORTED_AUDIO_FORMATS as frozensets. Do not use plain sets —
      frozensets are hashable and signal immutability.
    - The passthrough thresholds (TARGET_SAMPLE_RATE, TARGET_LUFS, etc.)
      are constants here, not CLI flags. They reflect pyannote's fixed
      requirements, not user preferences. Do not make them configurable.
    - run_stage1() is the only function external code should call.
      All other functions in this module are implementation details
      that run_stage1() coordinates.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import AudioIngestionError
from .hashing import compute_sha256
from .models import AudioProperties, LoudnessProperties
from .probe import measure_loudness, probe_audio_properties

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

#: Video container formats accepted as input.
#: Audio is extracted via ffmpeg -vn (discard video stream).
SUPPORTED_VIDEO_FORMATS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".avi", ".mkv"
})

#: Pure audio formats accepted as input.
SUPPORTED_AUDIO_FORMATS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".m4a", ".flac"
})

#: All supported formats — used for initial validation.
ALL_SUPPORTED_FORMATS: frozenset[str] = (
    SUPPORTED_VIDEO_FORMATS | SUPPORTED_AUDIO_FORMATS
)


# ---------------------------------------------------------------------------
# Normalization targets
#
# These values reflect pyannote's fixed requirements, not preferences.
# pyannote's models were trained at 16kHz mono. These are not configurable.
# ---------------------------------------------------------------------------

#: pyannote segmentation and embedding models require 16kHz input.
TARGET_SAMPLE_RATE: int = 16_000

#: pyannote expects mono audio.
TARGET_CHANNELS: int = 1

#: EBU R128 integrated loudness target applied during Stage 3.
TARGET_LUFS: float = -16.0

#: Tolerance around TARGET_LUFS for passthrough decisions.
#: Files within ±1 LUFS of -16.0 are considered adequately normalized.
LUFS_PASSTHROUGH_TOLERANCE: float = 1.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PassthroughDecision:
    """
    Result of comparing a file's audio properties against normalization
    targets. Determines whether Stage 3 normalization can be skipped.

    Attributes
    ----------
    can_skip_normalization : bool
        True only when all property checks passed. When True, Stage 3
        is skipped and the file is used as-is by downstream stages.
    reasons : list[str]
        Human-readable list of reasons normalization is needed.
        Empty when can_skip_normalization is True.
    checks_performed : dict[str, bool]
        Mapping of property name to whether it passed.
        Written to output metadata for full auditability — a reviewer
        can see exactly which checks ran and what they found.
    """
    can_skip_normalization: bool
    reasons: list[str] = field(default_factory=list)
    checks_performed: dict[str, bool] = field(default_factory=dict)


@dataclass
class Stage1Result:
    """
    Complete output of Stage 1: Input Handling & Format Detection.

    Passed directly to Stage 2 (Audio Quality Profiling) and also
    serialized into the processing_notes block of the final output JSON
    via to_processing_notes().

    Attributes
    ----------
    original_path : Path
        Resolved absolute path to the original input file.
    original_filename : str
        Basename of the original input file.
    original_format : str
        Lowercase file extension (e.g. ".mp4", ".wav").
    is_video_input : bool
        True if the input was a video container. Audio will be extracted
        via ffmpeg -vn during Stage 3 if normalization runs, or as a
        preprocessing step if passthrough applies.
    sha256 : str
        SHA256 hex digest of the original file, computed before any
        processing. This is the chain-of-custody provenance record.
    audio_properties : AudioProperties
        Raw audio stream properties from ffprobe.
    loudness : LoudnessProperties | None
        Loudness measurement from ffmpeg loudnorm analysis.
        None if the loudness check was skipped because earlier property
        checks (sample rate, channels, codec) already required normalization.
    passthrough : PassthroughDecision
        Whether normalization can be skipped and the reasons if not.
    """
    original_path: Path
    original_filename: str
    original_format: str
    is_video_input: bool
    sha256: str
    audio_properties: AudioProperties
    loudness: LoudnessProperties | None
    passthrough: PassthroughDecision

    def to_processing_notes(self) -> dict:
        """
        Serialize Stage 1 results into the processing_notes dict format
        expected by the TrustScript output JSON schema.

        Called by Stage 7 (Output Assembly, src/utils/schema.py) when
        building the final {incident_id}_phase1.json combined output.

        Returns
        -------
        dict
            Serialized Stage 1 metadata for inclusion in processing_notes.
        """
        return {
            "stage_1": {
                "original_filename": self.original_filename,
                "original_format": self.original_format,
                "is_video_input": self.is_video_input,
                "original_sha256": self.sha256,
                "passthrough": self.passthrough.can_skip_normalization,
                "passthrough_checks": self.passthrough.checks_performed,
                "normalization_required_reasons": self.passthrough.reasons,
                "audio_properties": {
                    "sample_rate_hz": self.audio_properties.sample_rate,
                    "channels": self.audio_properties.channels,
                    "codec": self.audio_properties.codec_name,
                    "duration_seconds": self.audio_properties.duration_seconds,
                    "bit_rate_bps": self.audio_properties.bit_rate,
                },
                "loudness_measurement": (
                    {
                        "integrated_lufs": self.loudness.integrated_lufs,
                        "loudness_range_lu": self.loudness.loudness_range,
                        "true_peak_dbtp": self.loudness.true_peak,
                    }
                    if self.loudness is not None
                    else None
                ),
            }
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_input_path(path: Path) -> None:
    """
    Confirm the input path exists, is a regular file, and has a supported
    extension.

    This is the first check in Stage 1. It runs before hashing or probing
    so that clear, actionable errors are raised early without wasted work.

    Parameters
    ----------
    path : Path
        Resolved absolute path to validate.

    Raises
    ------
    AudioIngestionError
        If the path does not exist, is a directory, or has an unsupported
        file extension.
    """
    if not path.exists():
        raise AudioIngestionError(
            f"Input file not found: {path}\n"
            "Verify the path is correct and the file is accessible."
        )

    if not path.is_file():
        raise AudioIngestionError(
            f"Input path is not a file: {path}\n"
            "To process a folder in batch mode, pass a directory path "
            "to the CLI's --input flag."
        )

    suffix = path.suffix.lower()
    if suffix not in ALL_SUPPORTED_FORMATS:
        raise AudioIngestionError(
            f"Unsupported file format: '{suffix}' ({path.name})\n"
            f"Supported video formats: {sorted(SUPPORTED_VIDEO_FORMATS)}\n"
            f"Supported audio formats: {sorted(SUPPORTED_AUDIO_FORMATS)}"
        )

    logger.debug("Input validation passed: %s", path.name)


# ---------------------------------------------------------------------------
# Passthrough decision
# ---------------------------------------------------------------------------


def check_passthrough(
    properties: AudioProperties,
    loudness: LoudnessProperties | None = None,
) -> PassthroughDecision:
    """
    Compare audio properties against normalization targets and decide
    whether Stage 3 can be skipped.

    A file qualifies for passthrough only when all four checks pass:

        1. Sample rate is exactly 16000 Hz.
        2. Audio is mono (1 channel).
        3. Codec is PCM (uncompressed WAV — pyannote cannot read compressed
           codecs like aac or mp3 without transcoding).
        4. Integrated loudness is within ±1 LUFS of -16.0 LUFS.

    Checks run in order. If checks 1–3 already fail, loudness is irrelevant
    (normalization is required regardless), so the caller should not bother
    calling measure_loudness() and should pass loudness=None.

    When loudness=None and checks 1–3 pass, the loudness check is marked
    as unverified and normalization is conservatively flagged as required.
    The run_stage1() orchestrator handles the conditional loudness call
    correctly — direct callers should do the same.

    Parameters
    ----------
    properties : AudioProperties
        Audio properties from probe_audio_properties().
    loudness : LoudnessProperties | None
        Loudness measurement from measure_loudness(). Required when
        checks 1–3 pass and the loudness decision matters. Pass None
        when normalization is already known to be required.

    Returns
    -------
    PassthroughDecision
        Whether normalization can be skipped, why not if not, and
        an auditability record of which checks were performed.

    Examples
    --------
    >>> # Already-normalized file:
    >>> props = AudioProperties(16000, 1, "pcm_s16le", 300.0)
    >>> loud = LoudnessProperties(-16.2, 5.1, -3.0)
    >>> decision = check_passthrough(props, loud)
    >>> decision.can_skip_normalization
    True

    >>> # Typical BWC camera output — needs normalization:
    >>> props = AudioProperties(48000, 2, "aac", 3612.0)
    >>> decision = check_passthrough(props)
    >>> decision.can_skip_normalization
    False
    >>> len(decision.reasons)
    3
    """
    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # --- Check 1: Sample rate ---
    sample_rate_ok = properties.sample_rate == TARGET_SAMPLE_RATE
    checks["sample_rate"] = sample_rate_ok
    if not sample_rate_ok:
        reasons.append(
            f"Sample rate is {properties.sample_rate} Hz "
            f"(target: {TARGET_SAMPLE_RATE} Hz)"
        )

    # --- Check 2: Channels ---
    channels_ok = properties.channels == TARGET_CHANNELS
    checks["channels"] = channels_ok
    if not channels_ok:
        reasons.append(
            f"Audio has {properties.channels} channels "
            f"(target: mono / {TARGET_CHANNELS} channel)"
        )

    # --- Check 3: Codec ---
    # pcm_s16le and pcm_s24le are both acceptable PCM variants.
    # aac, mp3, etc. require transcoding even at the correct sample rate.
    codec_ok = properties.codec_name.startswith("pcm_")
    checks["codec_pcm"] = codec_ok
    if not codec_ok:
        reasons.append(
            f"Codec is {properties.codec_name} — not PCM WAV "
            "(pyannote requires uncompressed PCM input)"
        )

    # --- Check 4: Loudness ---
    # Only meaningful when checks 1–3 pass. If normalization is already
    # required, the loudness check result is irrelevant.
    if reasons:
        # Earlier checks failed — loudness check skipped, not applicable.
        checks["loudness_lufs"] = False
    elif loudness is None:
        # Checks 1–3 passed but no loudness measurement was provided.
        # Conservatively flag normalization as required.
        checks["loudness_lufs"] = False
        reasons.append(
            "Loudness not measured — normalization assumed required. "
            "(run_stage1() handles this automatically; only relevant "
            "if calling check_passthrough() directly.)"
        )
    else:
        loudness_ok = (
            abs(loudness.integrated_lufs - TARGET_LUFS) <= LUFS_PASSTHROUGH_TOLERANCE
        )
        checks["loudness_lufs"] = loudness_ok
        if not loudness_ok:
            reasons.append(
                f"Integrated loudness is {loudness.integrated_lufs:.1f} LUFS "
                f"(target: {TARGET_LUFS} ± {LUFS_PASSTHROUGH_TOLERANCE} LUFS)"
            )

    can_skip = not reasons

    if can_skip:
        logger.info("Passthrough: all normalization targets met — Stage 3 skipped.")
    else:
        logger.info(
            "Normalization required (%d reason(s)): %s",
            len(reasons),
            "; ".join(reasons),
        )

    return PassthroughDecision(
        can_skip_normalization=can_skip,
        reasons=reasons,
        checks_performed=checks,
    )


# ---------------------------------------------------------------------------
# Stage 1 entry point
# ---------------------------------------------------------------------------


def run_stage1(path: Path) -> Stage1Result:
    """
    Execute Stage 1: Input Handling & Format Detection.

    This is the public entry point for Stage 1. External code (engine.py,
    cli.py) should call this function and pass the returned Stage1Result
    to run_stage2().

    Execution order
    ---------------
    1. Resolve path to absolute.
    2. Validate input path and extension.
    3. Compute SHA256 hash of the original file.
    4. Probe audio properties via ffprobe (fast, no decoding).
    5. Conditionally measure loudness via ffmpeg loudnorm:
           - Only runs if checks 1–3 (sample rate, channels, codec) pass.
           - Skipped if those checks already require normalization —
             the full-decode loudness pass is expensive and unnecessary
             when normalization is already known to be required.
    6. Make the passthrough decision.
    7. Return a complete Stage1Result.

    Parameters
    ----------
    path : Path
        Path to the input file. May be relative or absolute.
        Resolved to absolute at the start of execution.

    Returns
    -------
    Stage1Result
        Complete Stage 1 results. Pass to run_stage2() to continue.

    Raises
    ------
    AudioIngestionError
        If the file is missing, unsupported, or unreadable.
    FFprobeError
        If ffprobe or ffmpeg is not installed or fails on the file.

    Examples
    --------
    >>> from pathlib import Path
    >>> from src.utils.audio import run_stage1
    >>>
    >>> result = run_stage1(Path("footage/incident_001.mp4"))
    >>> result.sha256
    'a3f9c2d1...'
    >>> result.is_video_input
    True
    >>> result.passthrough.can_skip_normalization
    False
    >>> result.to_processing_notes()
    {'stage_1': {'original_filename': 'incident_001.mp4', ...}}
    """
    path = path.resolve()

    logger.info("Stage 1 — Input Handling & Format Detection: %s", path.name)

    # Step 1: Validate before any expensive operations.
    validate_input_path(path)

    suffix = path.suffix.lower()
    is_video = suffix in SUPPORTED_VIDEO_FORMATS
    logger.debug(
        "%s identified as %s.",
        path.name,
        "video container (audio will be extracted)" if is_video else "audio file",
    )

    # Step 2: Hash the original file. Must happen before anything else.
    sha256 = compute_sha256(path)

    # Step 3: Probe audio properties (fast — no audio decoding).
    audio_props = probe_audio_properties(path)

    # Step 4: Conditionally measure loudness.
    # The loudnorm analysis pass requires a full decode of the audio stream.
    # On a 60-minute BWC file at 48kHz this can take 30–90 seconds on CPU.
    # Only run it if the cheaper checks already pass, making loudness the
    # only remaining question for the passthrough decision.
    sample_rate_ok = audio_props.sample_rate == TARGET_SAMPLE_RATE
    channels_ok = audio_props.channels == TARGET_CHANNELS
    codec_ok = audio_props.codec_name.startswith("pcm_")

    loudness: LoudnessProperties | None = None
    if sample_rate_ok and channels_ok and codec_ok:
        logger.debug(
            "Sample rate, channels, codec all pass — "
            "running loudness measurement for passthrough decision."
        )
        loudness = measure_loudness(path)
    else:
        logger.debug(
            "Skipping loudness measurement — "
            "earlier checks already require normalization."
        )

    # Step 5: Make the passthrough decision from all collected properties.
    passthrough = check_passthrough(audio_props, loudness)

    result = Stage1Result(
        original_path=path,
        original_filename=path.name,
        original_format=suffix,
        is_video_input=is_video,
        sha256=sha256,
        audio_properties=audio_props,
        loudness=loudness,
        passthrough=passthrough,
    )

    logger.info(
        "Stage 1 complete — %s | sha256: %s... | passthrough=%s",
        path.name,
        sha256[:12],
        passthrough.can_skip_normalization,
    )

    return result