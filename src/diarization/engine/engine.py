"""
src/diarization/engine/engine.py
===================================
TrustScript Phase 1 — Stage 5: Windowed Diarization Orchestrator

Responsibility
--------------
Top-level orchestration only. Two functions:

    run_windowed_diarization()
        Entry point. Reads audio metadata, initialises overlap class
        indices, generates chunk windows, and drives the chunk loop
        under torch.no_grad(). Collects timeline segments and speech
        intervals. Returns EngineResult.

    _process_chunk()
        Per-chunk coordinator. Loads audio, runs pyannote, unwraps
        the pipeline output, extracts posteriors and overlap timeline,
        computes the core window, then calls Pass 1 → candidate
        preparation → Pass 2 in order.

No extraction logic lives here. All per-segment work is delegated to:
    audio.py        — waveform I/O and embeddings
    posteriors.py   — powerset posterior extraction
    overlap.py      — overlap detection
    candidates.py   — Gate 3 pool building and conflict resolution
    passes.py       — the two-pass segment loop

Hardware guardrail
------------------
torch.no_grad() wraps the entire chunk loop in run_windowed_diarization.

Without this, PyTorch builds a backpropagation graph for every forward
pass. On an 8GB machine processing 5-minute chunks, this exhausts RAM
within the first two chunks and produces an OOM crash with no useful
error message.

torch.no_grad() is applied at the outermost loop level — not per-call —
to cover torchaudio loads, pyannote inference, and embedding extraction
within a single context.

PCM WAV contract
----------------
audio_path MUST be the Stage 3 normalized output: a pcm_s16le mono WAV.
See audio.py module docstring for the full explanation of why compressed
formats cause silent full-file RAM spikes that bypass all other guards.

DiarizeOutput compatibility
---------------------------
pyannote/speaker-diarization-community-1 returns a DiarizeOutput
dataclass rather than a bare pyannote.core.Annotation. All downstream
code (itertracks, get_overlap, crop) expects a bare Annotation.
_process_chunk unwraps the pipeline result immediately after the call
via hasattr checks, handling both the community model format and the
older bare-Annotation format without breaking either path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
import torch

from ..models import Stage4Result
from ..vault import SpeakerVault
from .audio import load_chunk_audio
from .candidates import build_candidates_by_label, resolve_local_conflicts
from .chunking import compute_core_window, generate_chunk_windows
from .passes import collect_raw_segments, run_vault_matching
from .posteriors import extract_posteriors, get_overlap_class_indices
from .types import DEFAULT_CHUNK_SIZE, EngineResult

if TYPE_CHECKING:
    from pyannote.core import Annotation, SlidingWindowFeature, Timeline

logger = logging.getLogger(__name__)


def run_windowed_diarization(
    audio_path: Path,
    stage4_result: Stage4Result,
    vault: SpeakerVault,
    chunk_size: float = DEFAULT_CHUNK_SIZE,
) -> EngineResult:
    """
    Run windowed diarization over the full normalized audio file.

    Processes audio in overlapping chunks. For each chunk: runs pyannote,
    extracts embeddings and concurrency_confidence, and matches against
    the Speaker Anchor Vault. Returns the complete timeline and populated
    vault.

    Parameters
    ----------
    audio_path : Path
        Stage 3 normalized pcm_s16le mono WAV. See audio.py for the
        PCM WAV contract — passing a compressed file causes silent
        full-file RAM spikes that torch.no_grad() cannot prevent.
    stage4_result : Stage4Result
        Loaded pyannote models from Stage 4.
    vault : SpeakerVault
        Live vault, possibly pre-seeded by stage5.py initialization logic.
        Mutated in place throughout the chunk loop.
    chunk_size : float
        Chunk duration in seconds. Default: 300.0 (5 minutes).

    Returns
    -------
    EngineResult
        Complete timeline, populated vault, and VAD speech intervals.
    """
    pipeline = stage4_result.pipeline
    embed_inference = stage4_result.embed_inference

    # Read audio metadata using soundfile — more reliable than torchaudio.info()
    # across CPU-only wheel versions. The file is guaranteed to be a pcm_s16le
    # mono WAV from Stage 3, which soundfile reads without backend dependencies.
    sf_info = sf.info(str(audio_path))
    sample_rate: int = sf_info.samplerate
    audio_duration: float = sf_info.frames / sf_info.samplerate

    logger.info(
        "Engine: audio=%.2fs @ %dHz, chunk=%.0fs, overlap=10s",
        audio_duration, sample_rate, chunk_size,
    )

    # Compute powerset overlap class indices once before the loop.
    # Returns [] on failure — engine degrades gracefully (confidence = None).
    overlap_class_indices = get_overlap_class_indices(pipeline)
    if not overlap_class_indices:
        logger.warning(
            "Powerset overlap class indices unavailable. "
            "concurrency_confidence will be None for all segments."
        )

    chunk_windows = generate_chunk_windows(audio_duration, chunk_size)
    num_chunks = len(chunk_windows)

    timeline: list = []
    speech_intervals: list[tuple[float, float]] = []
    segment_counter: int = 0

    # ── Hardware guardrail ────────────────────────────────────────────────
    # torch.no_grad() covers the entire loop — see module docstring.
    # ─────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_windows):
            is_first = chunk_idx == 0
            is_last = chunk_idx == num_chunks - 1

            logger.info(
                "Chunk %d/%d: %.1fs – %.1fs",
                chunk_idx + 1, num_chunks, chunk_start, chunk_end,
            )

            chunk_segments, chunk_speech = _process_chunk(
                audio_path=audio_path,
                sample_rate=sample_rate,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                chunk_idx=chunk_idx,
                is_first=is_first,
                is_last=is_last,
                pipeline=pipeline,
                embed_inference=embed_inference,
                vault=vault,
                overlap_class_indices=overlap_class_indices,
                segment_id_offset=segment_counter,
            )

            timeline.extend(chunk_segments)
            speech_intervals.extend(chunk_speech)
            segment_counter += len(chunk_segments)

    total_speech = sum(e - s for s, e in speech_intervals)
    logger.info(
        "Engine complete: %d segments, %.2fs speech, %d chunks.",
        len(timeline), total_speech, num_chunks,
    )

    return EngineResult(
        timeline=timeline,
        vault=vault,
        chunk_count=num_chunks,
        total_speech_seconds=total_speech,
        speech_intervals=speech_intervals,
    )


def _process_chunk(
    audio_path: Path,
    sample_rate: int,
    chunk_start: float,
    chunk_end: float,
    chunk_idx: int,
    is_first: bool,
    is_last: bool,
    pipeline,
    embed_inference,
    vault: SpeakerVault,
    overlap_class_indices: list[int],
    segment_id_offset: int,
) -> tuple[list, list[tuple[float, float]]]:
    """
    Full processing pipeline for one chunk.

    Sequence:
        1.  Load chunk audio (PCM WAV random access — see audio.py).
        2.  Run pyannote diarization.
        2b. Unwrap DiarizeOutput → bare Annotation (community model compat).
        3.  Extract powerset posteriors (second forward pass).
        4.  Get precomputed overlap timeline from pyannote.
        5.  Compute core window for seam management.
        6.  Pass 1: collect_raw_segments().
        7.  Build Gate 3 candidate pools.
        8.  Resolve local speaker conflicts.
        9.  Pass 2: run_vault_matching().
        10. Extract speech intervals for refined SNR.

    Returns
    -------
    tuple[list[TimelineSegment], list[tuple[float, float]]]
        (accepted segments, speech intervals as (start_sec, end_sec))
    """
    # Step 1 — Audio load.
    chunk_waveform = load_chunk_audio(audio_path, chunk_start, chunk_end, sample_rate)
    audio_input = {"waveform": chunk_waveform, "sample_rate": sample_rate}

    # Step 2 — Diarization.
    # Blocking CPU call — 2–10 min per chunk on ARM depending on chunk size.
    # No progress feedback from pyannote during this call.
    logger.info(
        "Chunk %d: running pyannote inference (%.0fs of audio) — "
        "this may take several minutes on CPU.",
        chunk_idx, chunk_end - chunk_start,
    )
    try:
        result = pipeline(audio_input)
    except Exception as exc:
        logger.error(
            "Chunk %d: pyannote diarization failed: %s — skipping chunk.",
            chunk_idx, exc,
        )
        return [], []

    # Step 2b — Unwrap DiarizeOutput → bare Annotation.
    # pyannote/speaker-diarization-community-1 returns a DiarizeOutput
    # dataclass rather than a bare Annotation. itertracks(), get_overlap(),
    # and crop() all live on the bare Annotation — unwrap before use.
    # Older model versions return the Annotation directly; the hasattr
    # fallback handles both without breaking either path.
    # if hasattr(result, "diarization"):
    #     diarization: "Annotation" = result.diarization
    # elif hasattr(result, "annotation"):
    #     diarization = result.annotation
    # else:
    #     diarization = result  # bare Annotation — older pipeline format
        
    # Step 2b — Unwrap DiarizeOutput → bare Annotation.
    if hasattr(result, "speaker_diarization"):       # <--- FIX: "speaker_diarization"
        diarization: "Annotation" = result.speaker_diarization
    elif hasattr(result, "annotation"):
        diarization = result.annotation
    else:
        diarization = result  # bare Annotation — older pipeline format

    # Step 3 — Powerset posteriors (internal API — see posteriors.py).
    posteriors: "SlidingWindowFeature | None" = extract_posteriors(pipeline, audio_input)

    # Step 4 — Overlap timeline.
    overlap_timeline: "Timeline | None" = None
    try:
        overlap_timeline = diarization.get_overlap()
    except Exception as exc:
        logger.warning(
            "Chunk %d: diarization.get_overlap() failed: %s. "
            "Overlap detection will use intersection fallback.",
            chunk_idx, exc,
        )

    # Step 5 — Core window.
    core_start, core_end = compute_core_window(
        chunk_start, chunk_end, is_first, is_last
    )
    logger.debug(
        "Chunk %d: core window [%.2f, %.2f]", chunk_idx, core_start, core_end
    )

    # Step 6 — Pass 1.
    raw_segments = collect_raw_segments(
        diarization=diarization,
        chunk_waveform=chunk_waveform,
        sample_rate=sample_rate,
        chunk_start=chunk_start,
        chunk_idx=chunk_idx,
        core_start=core_start,
        core_end=core_end,
        overlap_timeline=overlap_timeline,
        posteriors=posteriors,
        overlap_class_indices=overlap_class_indices,
        embed_inference=embed_inference,
        segment_id_offset=segment_id_offset,
    )

    if not raw_segments:
        logger.debug("Chunk %d: no segments in core window.", chunk_idx)
        return [], []

    # Step 7 — Gate 3 candidate pools.
    candidates_by_label = build_candidates_by_label(raw_segments)

    # Step 8 — Conflict resolution.
    candidates_by_label = resolve_local_conflicts(
        raw_segments, candidates_by_label, vault
    )

    # Step 9 — Pass 2: vault matching.
    matched_segments = run_vault_matching(
        raw_segments=raw_segments,
        candidates_by_label=candidates_by_label,
        vault=vault,
    )

    # Step 10 — Speech intervals for refined SNR (float seconds).
    # stage5._compute_refined_snr() converts to sample offsets internally
    # using int(sec * sample_rate) — no precision issue at the SNR calc.
    speech_intervals = [
        (seg.start_seconds, seg.end_seconds)
        for seg in matched_segments
        if not seg.overlap
    ]

    return matched_segments, speech_intervals