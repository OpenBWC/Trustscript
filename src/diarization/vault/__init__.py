"""
src/diarization/vault/__init__.py
===================================
Public interface for the vault package.

External callers (engine.py, stage5.py, tests) import from here.
Internal vault modules import from each other directly.

Exports
-------
SpeakerVault
    The main class. Import this everywhere you need vault operations.

MATCH_THRESHOLD
    Cosine similarity threshold for matching. Exported so engine.py
    and tests can reference it without importing vault/types.py directly.

SPEAKER_ID_PREFIX
    "TRUST_SPK" — exported for any code that needs to parse or
    construct global speaker IDs.

HIGH_VARIANCE_THRESHOLD
    Std_dev threshold above which HIGH_VARIANCE_SPEAKER is flagged.
    Exported for tests that assert on vault metadata output.
"""

from .types import HIGH_VARIANCE_THRESHOLD, MATCH_THRESHOLD, SPEAKER_ID_PREFIX
from .vault import SpeakerVault

__all__ = [
    "SpeakerVault",
    "MATCH_THRESHOLD",
    "SPEAKER_ID_PREFIX",
    "HIGH_VARIANCE_THRESHOLD",
]