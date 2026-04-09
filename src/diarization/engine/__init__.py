"""
src/diarization/engine/__init__.py
=====================================
Public interface for the engine package.

External callers (stage5.py, tests) import from here.
Internal engine modules import from each other directly.

Exports
-------
run_windowed_diarization
    Primary entry point. Called by stage5.py after vault initialization.

EngineResult
    Return type of run_windowed_diarization(). Consumed by stage5.py
    for timeline assembly and refined SNR computation.

DEFAULT_CHUNK_SIZE
    300.0 seconds. Exported so stage5.py and the CLI can reference
    the production default without importing engine/types.py directly.
"""

from .engine import run_windowed_diarization
from .types import DEFAULT_CHUNK_SIZE, EngineResult

__all__ = [
    "run_windowed_diarization",
    "EngineResult",
    "DEFAULT_CHUNK_SIZE",
]