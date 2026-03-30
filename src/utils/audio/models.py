"""
src/utils/audio/models.py
==========================
TrustScript Phase 1 — Shared Audio Data Models

Responsibility
--------------
Dataclasses that represent raw audio measurements. These are plain
data containers with no logic. They exist so that hashing.py,
probe.py, and stage1.py can share structured types without any of
those modules importing from each other.

As later stages are added (Stage 2: Quality Profiling, Stage 3:
Normalization), their output dataclasses should be added here too.

Contributors
------------
    Keep this module free of logic. Dataclasses only. Any function
    that computes, transforms, or validates belongs in the module
    that owns that stage.
"""

from dataclasses import dataclass


@dataclass
class AudioProperties:
    """
    Raw audio stream properties extracted from an input file via ffprobe.

    These reflect the file's actual state before any normalization.
    All Stage 1 decision logic reads from this dataclass rather than
    re-probing the file.

    Attributes
    ----------
    sample_rate : int
        Audio sample rate in Hz (e.g. 48000, 44100, 16000).
    channels : int
        Number of audio channels (1 = mono, 2 = stereo).
    codec_name : str
        Audio codec as reported by ffprobe (e.g. "pcm_s16le", "aac", "mp3").
    duration_seconds : float
        Total duration of the audio stream in seconds.
    bit_rate : int | None
        Audio bit rate in bits per second. None when ffprobe cannot
        determine bit rate without a full decode (e.g. some lossless formats).
    """
    sample_rate: int
    channels: int
    codec_name: str
    duration_seconds: float
    bit_rate: int | None = None


@dataclass
class LoudnessProperties:
    """
    Integrated loudness measurement from an ffmpeg loudnorm analysis pass.

    Produced by probe.measure_loudness(). Requires a full decode of the
    audio content, so it is only computed when needed for the passthrough
    decision (see stage1.run_stage1() for the conditional logic).

    Attributes
    ----------
    integrated_lufs : float
        Integrated loudness in LUFS (Loudness Units Full Scale).
        TrustScript's Stage 3 target is -16.0 LUFS per EBU R128.
    loudness_range : float
        Loudness Range (LRA) in LU. High LRA indicates highly dynamic
        audio (e.g. quiet speech followed by shouting). Informational only —
        not used in passthrough decisions but written to output metadata.
    true_peak : float
        Maximum true peak level in dBTP (decibels True Peak).
        Values above 0.0 indicate digital clipping. Used as a secondary
        clipping indicator in Stage 2 Audio Quality Profiling.
    """
    integrated_lufs: float
    loudness_range: float
    true_peak: float