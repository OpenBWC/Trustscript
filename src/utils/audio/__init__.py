"""
src/utils/audio/__init__.py
============================
TrustScript Phase 1 — Audio Package Public API

This package handles all audio ingestion for the TrustScript pipeline.
External code should import only from here, not from submodules directly.

    from src.utils.audio import run_stage1, Stage1Result

This keeps the internal file structure an implementation detail.
If modules are renamed or split in the future, imports across the
codebase don't need to change, only this file does.

Exposed API
-----------
Stage execution:
    run_stage1          Entry point for Stage 1: Input Handling & Detection.

Result types:
    Stage1Result        Complete Stage 1 output, passed to Stage 2.
    PassthroughDecision Whether normalization was skipped and why.
    AudioProperties     Raw audio stream properties from ffprobe.
    LoudnessProperties  Integrated loudness from ffmpeg loudnorm analysis.

Exceptions:
    AudioIngestionError Base exception for all audio ingestion errors.
    FFprobeError        ffprobe/ffmpeg failure or missing installation.

Stages not yet implemented (added here as they are built):
    run_stage2          Audio Quality Profiling  (Stage 2)
    run_stage3          Normalization            (Stage 3)
"""

from .exceptions import AudioIngestionError, FFprobeError
from .models import AudioProperties, LoudnessProperties
from .stage1 import PassthroughDecision, Stage1Result, run_stage1

__all__ = [
    # Stage entry points
    "run_stage1",
    # Result dataclasses
    "Stage1Result",
    "PassthroughDecision",
    "AudioProperties",
    "LoudnessProperties",
    # Exceptions
    "AudioIngestionError",
    "FFprobeError",
]