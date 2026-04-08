"""
src/diarization/engine/audio.py
==================================
Audio I/O and embedding extraction.

Responsibility
--------------
Three functions that touch audio data directly:

    load_chunk_audio()
        Loads one chunk from disk as a (1, num_samples) float32 tensor
        using torchaudio's frame_offset random access.

    extract_segment_embedding()
        Slices a segment from the chunk waveform and runs ECAPA-TDNN
        inference via pyannote's Inference object. Applies two
        zero-vector guards to prevent NaN propagation downstream.

    compute_rms()
        Computes RMS energy for a segment waveform slice.
        Returns float | None — 0.0 is a valid value (digital silence).

PCM WAV contract
----------------
audio_path MUST be the Stage 3 normalized output: a pcm_s16le mono
WAV file.

torchaudio.load() with frame_offset on an uncompressed PCM WAV
resolves to a true byte-level fseek() — only requested samples are
decoded from disk.

On compressed / container formats (MP4, M4A, AAC, MP3), the
underlying sox_io / ffmpeg backend decodes the entire file from the
start up to (frame_offset + num_frames) before slicing. On chunk 10
of a 3-hour recording this silently allocates ~2.5 hours of decoded
audio into RAM. torch.no_grad() provides no protection against this —
it is a C-extension allocation, not a PyTorch tensor graph. The
machine OOMs without a useful error message.

Stage 3 normalization exists in part to prevent this failure.
Do not bypass Stage 3 or pass raw MP4/M4A files to the engine.

Zero-vector defence
-------------------
_extract_segment_embedding applies two guards:
    1. Before inference: skip if waveform norm < ZERO_VECTOR_THRESHOLD.
       Catches silent / corrupted frames before spending GPU/CPU time.
    2. After inference:  skip if embedding norm < ZERO_VECTOR_THRESHOLD.
       Catches pyannote glitches that produce zero-vector outputs
       despite non-silent input.
Both return None — the caller skips the segment entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchaudio

from .types import EMBEDDING_DTYPE, ZERO_VECTOR_THRESHOLD

if TYPE_CHECKING:
    from pyannote.audio import Inference

logger = logging.getLogger(__name__)


def load_chunk_audio(
    audio_path: Path,
    chunk_start: float,
    chunk_end: float,
    sample_rate: int,
) -> torch.Tensor:
    """
    Load a chunk of audio as a mono (1, num_samples) float32 tensor.

    Uses torchaudio's frame_offset + num_frames for efficient random
    access. See module docstring for the PCM WAV contract — passing a
    compressed file causes a silent full-file decode into RAM.

    If the source file is multi-channel (unexpected after Stage 3
    normalization), channels are averaged to mono with a warning.
    Sample rate mismatches trigger a warning and resample.

    Parameters
    ----------
    audio_path : Path
        Stage 3 normalized pcm_s16le mono WAV.
    chunk_start, chunk_end : float
        Absolute file-time boundaries in seconds.
    sample_rate : int
        Expected sample rate (from torchaudio.info at startup).

    Returns
    -------
    torch.Tensor
        Shape (1, num_samples), dtype float32.
    """
    frame_offset = int(chunk_start * sample_rate)
    num_frames = int((chunk_end - chunk_start) * sample_rate)

    waveform, sr = torchaudio.load(
        str(audio_path),
        frame_offset=frame_offset,
        num_frames=num_frames,
    )

    if waveform.shape[0] > 1:
        raise RuntimeError(
            f"load_chunk_audio received {waveform.shape[0]}-channel audio. "
            f"Stage 5 requires a mono pcm_s16le WAV produced by Stage 3 "
            f"normalization. Multi-channel input means Stage 3 was skipped "
            f"or its output was replaced. Averaging to mono here would "
            f"constitute undocumented signal modification not captured in "
            f"processing_notes, invalidating the forensic chain of custody. "
            f"Re-run Stage 3 on the source file before proceeding."
        )

    if sr != sample_rate:
        raise RuntimeError(
            f"load_chunk_audio: sample rate mismatch — expected {sample_rate}Hz, "
            f"got {sr}Hz. Stage 5 requires audio normalized to a fixed sample "
            f"rate by Stage 3. Resampling here would introduce interpolation "
            f"artifacts not captured in processing_notes, invalidating the "
            f"forensic chain of custody. Re-run Stage 3 on the source file."
        )

    return waveform


def extract_segment_embedding(
    embed_inference: "Inference",
    chunk_waveform: torch.Tensor,
    sample_rate: int,
    seg_start_local: float,
    seg_end_local: float,
) -> np.ndarray | None:
    """
    Extract a 512-d ECAPA-TDNN embedding for one segment.

    Uses chunk-local times (seconds from chunk start, not file start).
    Applies two zero-vector guards — see module docstring.

    Handles both (512,) and (1, 512) output shapes from Inference.
    Returns None on any failure; caller skips the segment.

    Parameters
    ----------
    embed_inference : Inference
        pyannote Inference object from Stage 4 result.
    chunk_waveform : torch.Tensor
        Full chunk waveform, shape (1, num_chunk_samples).
    sample_rate : int
        Audio sample rate in Hz.
    seg_start_local, seg_end_local : float
        Segment boundaries in chunk-local time (seconds from chunk start).

    Returns
    -------
    np.ndarray | None
        512-d float32 embedding, or None if extraction failed.
    """
    frame_start = int(seg_start_local * sample_rate)
    frame_end = min(int(seg_end_local * sample_rate), chunk_waveform.shape[1])

    if frame_end <= frame_start:
        return None

    seg_waveform = chunk_waveform[:, frame_start:frame_end]

    # Guard 1 — waveform zero-vector.
    waveform_norm = float(torch.linalg.norm(seg_waveform))
    if waveform_norm < ZERO_VECTOR_THRESHOLD:
        logger.debug(
            "Segment [%.2fs, %.2fs]: near-zero waveform (norm=%.2e) — skipping.",
            seg_start_local, seg_end_local, waveform_norm,
        )
        return None

    try:
        audio_input = {"waveform": seg_waveform, "sample_rate": sample_rate}
        raw = embed_inference(audio_input)

        embedding: np.ndarray = np.asarray(raw, dtype=EMBEDDING_DTYPE)
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)

        # Guard 2 — embedding zero-vector.
        if np.linalg.norm(embedding) < ZERO_VECTOR_THRESHOLD:
            logger.warning(
                "Segment [%.2fs, %.2fs]: embed_inference returned "
                "near-zero embedding.",
                seg_start_local, seg_end_local,
            )
            return None

        return embedding

    except Exception:
        logger.exception(
            "Embedding extraction failed [%.2fs, %.2fs] — full traceback above. "
            "Common causes: pyannote shape mismatch, OOM on CPU, dtype error "
            "in upstream slicing code.",
            seg_start_local, seg_end_local,
        )
        return None


def compute_rms(
    chunk_waveform: torch.Tensor,
    sample_rate: int,
    seg_start_local: float,
    seg_end_local: float,
) -> float | None:
    """
    Compute RMS energy for a segment waveform slice.

    Returns float | None. 0.0 is a valid return value representing
    digital silence — callers must use `if rms is not None`, not
    `if rms`, when deciding whether to store the result.

    Parameters
    ----------
    chunk_waveform : torch.Tensor
        Full chunk waveform, shape (1, num_chunk_samples).
    sample_rate : int
        Audio sample rate in Hz.
    seg_start_local, seg_end_local : float
        Segment boundaries in chunk-local time.

    Returns
    -------
    float | None
        RMS energy, or None if the slice is empty or computation fails.
    """
    try:
        frame_start = int(seg_start_local * sample_rate)
        frame_end = min(int(seg_end_local * sample_rate), chunk_waveform.shape[1])
        if frame_end <= frame_start:
            return None
        seg_waveform = chunk_waveform[:, frame_start:frame_end]
        return float(torch.sqrt(torch.mean(seg_waveform ** 2)).item())
    except Exception as exc:
        logger.debug("RMS computation failed: %s", exc)
        return None