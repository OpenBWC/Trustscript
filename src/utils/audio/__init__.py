"""
src/utils/audio/__init__.py
============================
TrustScript Phase 1 — Audio Package Public API

This package handles all audio ingestion and quality profiling for the
TrustScript pipeline. External code should import only from here, not
from submodules directly.

    from src.utils.audio import run_stage1, run_stage2

This keeps the internal file structure an implementation detail.
If modules are renamed or split in the future, imports across the
codebase don't need to change — only this file does.

Exposed API
-----------
Stage execution:
    run_stage1          Stage 1: Input Handling & Format Detection
    run_stage2          Stage 2: Audio Quality Profiling

Result types:
    Stage1Result        Complete Stage 1 output.
    PassthroughDecision Whether normalization was skipped and why.
    Stage2Result        Complete Stage 2 output.
    AudioQualityProfile Audio quality measurements (SNR, clipping, etc.)
    AudioProperties     Raw audio stream properties from ffprobe.
    LoudnessProperties  Integrated loudness from ffmpeg loudnorm.

Constants:
    ALL_SUPPORTED_FORMATS   All supported input file extensions.

Exceptions:
    AudioIngestionError Base exception for all audio ingestion errors.
    FFprobeError        ffprobe/ffmpeg failure or missing installation.

Stages not yet implemented (added here as they are built):
    run_stage3          Normalization  (Stage 3)
"""

from .exceptions import AudioIngestionError, FFprobeError
from .models import (
    AudioProperties,
    AudioQualityProfile,
    LoudnessProperties,
    Stage2Result,
)
from .profiling import run_stage2
from .stage1 import (
    ALL_SUPPORTED_FORMATS,
    PassthroughDecision,
    Stage1Result,
    run_stage1,
)

__all__ = [
    # Stage entry points
    "run_stage1",
    "run_stage2",
    # Stage 1 result types
    "Stage1Result",
    "PassthroughDecision",
    # Stage 2 result types
    "Stage2Result",
    "AudioQualityProfile",
    # Shared model types
    "AudioProperties",
    "LoudnessProperties",
    # Constants
    "ALL_SUPPORTED_FORMATS",
    # Exceptions
    "AudioIngestionError",
    "FFprobeError",
]