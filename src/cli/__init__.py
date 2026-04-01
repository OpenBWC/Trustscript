"""
src/cli/__init__.py
====================
TrustScript CLI — Package Entry Point

This file is the only thing pyproject.toml needs to reference.
It imports the Click group from main.py and registers every subcommand
onto it. External code and pyproject.toml import `trustscript` from here.

    [project.scripts]
    trustscript = "src.cli:trustscript"

Adding a new subcommand
-----------------------
1. Create src/cli/{phase}.py with a Click command function.
2. Import it here and call trustscript.add_command(command_fn).
3. That's it — the command appears in `trustscript --help` automatically.

Contributors
------------
    Do not add CLI logic to this file. Logic belongs in the individual
    command files. This file is registration only.
"""

from .main import trustscript

# Phase 1 — Speaker Diarization Engine (implemented)
from .diarize import diarize

# Phase 2 — Overlap Characterization (stub)
from .overlap import overlap

# Phase 3 — ASR (stub)
from .transcribe import transcribe

# Phase 4 — Confidence Fusion (stub)
from .fuse import fuse

# Phase 5 — Output Package (stub)
from .package import package_output

# Full pipeline runner (stub)
from .run import run_pipeline

# Register all subcommands onto the main group.
trustscript.add_command(diarize)
trustscript.add_command(overlap)
trustscript.add_command(transcribe)
trustscript.add_command(fuse)
trustscript.add_command(package_output, name="package")
trustscript.add_command(run_pipeline, name="run")

__all__ = ["trustscript"]