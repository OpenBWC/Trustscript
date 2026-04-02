"""
src/utils/audio/__init__.py
============================
TrustScript Phase 1 — Audio Package Public API

External code imports only from here. Internal module structure is
an implementation detail that this file abstracts away.

    from src.utils.audio import run_stage1, extract_working_audio, run_stage2

Exposed API
-----------
Stage execution:
    run_stage1              Stage 1: Input Handling & Format Detection
    extract_working_audio   Extraction: video/compressed → working WAV
    run_stage2              Stage 2: Audio Quality Profiling

Stage 1 result types:
    Stage1Result
    PassthroughDecision

Stage 2 result types:
    Stage2Result
    AudioQualityProfile
    QualityWindow
    QualityEvent

Shared model types:
    AudioProperties
    LoudnessProperties

Constants:
    ALL_SUPPORTED_FORMATS
    FORMATS_NEEDING_EXTRACTION
    FORMATS_SOUNDFILE_NATIVE

Exceptions:
    AudioIngestionError
    FFprobeError

Stages not yet implemented (added here as they are built):
    run_stage3          Normalization  (Stage 3)
"""

from .exceptions import AudioIngestionError, FFprobeError
from .extraction import (
    FORMATS_NEEDING_EXTRACTION,
    FORMATS_SOUNDFILE_NATIVE,
    extract_working_audio,
)
from .models import (
    AudioProperties,
    AudioQualityProfile,
    LoudnessProperties,
    QualityEvent,
    QualityWindow,
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
    "extract_working_audio",
    "run_stage2",
    # Stage 1 result types
    "Stage1Result",
    "PassthroughDecision",
    # Stage 2 result types
    "Stage2Result",
    "AudioQualityProfile",
    "QualityWindow",
    "QualityEvent",
    # Shared model types
    "AudioProperties",
    "LoudnessProperties",
    # Constants
    "ALL_SUPPORTED_FORMATS",
    "FORMATS_NEEDING_EXTRACTION",
    "FORMATS_SOUNDFILE_NATIVE",
    # Exceptions
    "AudioIngestionError",
    "FFprobeError",
]