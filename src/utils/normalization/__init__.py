"""
src/utils/normalization/__init__.py
=====================================
TrustScript Phase 1 — Normalization Package Public API

External code imports only from here.

    from src.utils.normalization import run_stage3, Stage3Result

Exposed API
-----------
    run_stage3      Stage 3: Normalization
    Stage3Result    Complete Stage 3 output
"""

from .stage3 import Stage3Result, run_stage3

__all__ = [
    "run_stage3",
    "Stage3Result",
]