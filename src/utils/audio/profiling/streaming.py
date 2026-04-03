"""
src/utils/audio/streaming.py
=============================
TrustScript Phase 1 — Stage 2 Audio Streaming Pass

Responsibility
--------------
Read the working PCM WAV via soundfile.blocks() and collect per-frame
RMS and peak amplitude arrays. This is the single I/O pass for Stage 2.
All subsequent computation in vad.py, snr.py, and events.py operates
on the arrays returned here — the file is never read a second time.

Memory design
-------------
soundfile.blocks() streams audio one frame at a time. At any moment,
only one 20ms frame (~7.5 KB at 48kHz stereo) is held as a numpy array.
The frame is processed immediately and the buffer is overwritten by the
next read. What accumulates in memory are the output arrays:

    rms_per_frame:   one float32 per frame ≈ 2.9 MB for 2 hours at 16kHz
    peak_per_frame:  one float32 per frame ≈ 2.9 MB for 2 hours at 16kHz

Compare to full waveform load: ~460 MB for the same file.

Stereo handling
---------------
soundfile returns shape (n_samples, n_channels) for multi-channel audio.
Each block is averaged across channels before RMS/peak computation:

    block.mean(axis=1)  →  (n_samples,)

This per-frame downmix uses one frame's worth of memory — no full-file
stereo-to-mono array is ever allocated.

Why soundfile and not ffmpeg Popen streaming?
----------------------------------------------
soundfile is purpose-built for audio I/O: faster than ffmpeg pipe for
WAV reads, better Python integration, and cleaner EOF handling. It is
used here instead of subprocess.Popen because extraction.py guarantees
a native PCM WAV on disk before Stage 2 runs. The working file is always
a format soundfile can read natively.

Contributors
------------
    This module owns only the I/O pass. Do not add SNR computation,
    VAD classification, or event detection logic here. Those belong
    in snr.py, vad.py, and events.py respectively.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import FRAME_DURATION_MS,

logger = logging.getLogger(__name__)


def stream_frame_arrays(
    working_audio_path: Path,
    original_sample_rate: int,
    frame_samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Stream the working WAV via soundfile.blocks() and return per-frame
    RMS and peak amplitude arrays.

    Parameters
    ----------
    working_audio_path : Path
        Working PCM f32le WAV produced by extraction.py.
    original_sample_rate : int
        Original sample rate from Stage 1 AudioProperties.
        Used for debug logging only.
    frame_samples : int
        Number of samples per frame, computed dynamically in stage2.py
        as ``int(sample_rate * FRAME_DURATION_MS / 1000.0)``.
        Must reflect the ACTUAL sample rate of the working file, not
        the 16kHz analysis constant — using FRAME_SAMPLES directly
        here would cause a time-stretch bug on non-16kHz files.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        ``(rms_per_frame, peak_per_frame)`` or ``None`` on failure.
    """
    rms_list: list[float] = []
    peak_list: list[float] = []

    try:
        for block in sf.blocks(
            working_audio_path,
            blocksize=frame_samples,
            overlap=0,
            dtype="float32",
        ):
            if block.size == 0:
                continue

            # Downmix to mono on the fly.
            # block.ndim == 1 for mono, 2 for stereo/multichannel.
            # block.mean(axis=1) uses one frame's worth of memory.
            if block.ndim == 2:
                block = block.mean(axis=1)

            rms_list.append(float(np.sqrt(np.mean(block ** 2))))
            peak_list.append(float(np.max(np.abs(block))))

    except sf.SoundFileError as e:
        logger.warning(
            "soundfile failed to read %s: %s — "
            "quality measurements unavailable.",
            working_audio_path.name, e,
        )
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error reading %s: %s — "
            "quality measurements unavailable.",
            working_audio_path.name, e,
        )
        return None

    if not rms_list:
        logger.warning(
            "No frames read from %s. "
            "File may contain no audio content.",
            working_audio_path.name,
        )
        return None

    rms_array = np.array(rms_list, dtype=np.float32)
    peak_array = np.array(peak_list, dtype=np.float32)

    logger.debug(
        "Streamed %s: %d frames (%.1f seconds at original rate)",
        working_audio_path.name,
        len(rms_list),
        len(rms_list) * frame_samples / original_sample_rate,
    )

    return rms_array, peak_array