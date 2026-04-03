"""
src/utils/audio/exceptions.py
===============================
TrustScript Phase 1 — Audio Package Exceptions

Responsibility
--------------
Custom exception classes for the src/utils/audio package.

Keeping exceptions in a dedicated module means that any file in the
package can import them without creating circular dependencies, and
external callers can catch specific exception types without importing
the full module that raises them.

Contributors
------------
    Add new exception classes here when a new failure mode needs its
    own type. Inherit from AudioIngestionError for errors that occur
    during the ingestion pipeline so callers can catch the base class
    when they don't need to distinguish sub-types.
"""


class AudioIngestionError(Exception):
    """
    Base exception for all errors raised during audio ingestion (Stage 1).

    Raised when Stage 1 cannot proceed with the given input. This covers:
        - Missing or unreadable input files.
        - Unsupported file formats.
        - I/O errors during hashing or file reading.

    Callers should catch this and surface it as a pipeline error rather
    than allowing it to propagate as an unhandled exception.
    """


class FFprobeError(AudioIngestionError):
    """
    Raised when ffprobe or ffmpeg fails or returns unexpected output.

    This typically indicates one of:
        - ffmpeg/ffprobe is not installed at the OS level.
        - The input file is corrupt or in an unexpected format.
        - ffprobe returned invalid JSON or no audio stream was found.
        - The loudnorm analysis pass failed to produce parseable output.

    See README.md for OS-level ffmpeg installation instructions.
    """