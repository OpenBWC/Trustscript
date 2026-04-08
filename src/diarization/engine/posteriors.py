"""
src/diarization/engine/posteriors.py
=======================================
Powerset posterior extraction and concurrency_confidence computation.

Responsibility
--------------
Three functions that access pyannote's internal segmentation model to
extract raw overlap probabilities:

    get_overlap_class_indices()
        Inspects pipeline._segmentation.model to find which powerset
        class indices represent concurrent speech. Called once before
        the chunk loop. Returns [] on failure — all downstream code
        degrades gracefully by setting concurrency_confidence = None.

    extract_posteriors()
        Calls pipeline._segmentation(audio_input) to get the raw
        softmax posterior matrix for the full chunk.
        Returns SlidingWindowFeature | None.

    concurrency_confidence_for_segment()
        Crops the posterior matrix to a segment's time window and
        computes the mean per-frame probability of concurrent speech
        across all overlap classes.

Internal API notice
-------------------
ALL three functions access pyannote.audio internals:
    pipeline._segmentation          (attribute)
    pipeline._segmentation.model    (attribute)
    pipeline._segmentation(...)     (callable — second forward pass)

These are not part of pyannote's public API and may change between
minor versions. They are the only route to the raw powerset posteriors
that power concurrency_confidence. If they break, all three functions
return None / [] and the engine continues without confidence scores.
Pinned to pyannote.audio >= 3.0 / Community-1 weights.

The second forward pass through _segmentation adds ~0.1s per chunk.
It is covered by torch.no_grad() in engine.py's outer loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyannote.core import SlidingWindowFeature

logger = logging.getLogger(__name__)


def get_overlap_class_indices(pipeline) -> list[int]:
    """
    Identify which powerset class indices represent concurrent speech.

    Powerset class layout for a model with N max speakers:
        Index 0:       silence  (0 active speakers)
        Index 1..N:    single-speaker classes (one per speaker slot)
        Index N+1..:   overlap combinations   (2+ speakers active)

    Tries two attribute paths for model introspection to handle minor
    version differences in pyannote.audio 3.x:
        model.specifications.classes  (preferred)
        model.powerset.num_speakers   (fallback)

    Parameters
    ----------
    pipeline
        Loaded pyannote Pipeline from Stage 4 result.

    Returns
    -------
    list[int]
        Overlap class indices. Empty list on any failure — callers
        set concurrency_confidence = None for the entire chunk.
    """
    try:
        model = pipeline._segmentation.model

        try:
            num_speakers = len(model.specifications.classes)
        except AttributeError:
            num_speakers = model.powerset.num_speakers

        try:
            num_classes = model.powerset.num_powerset_classes
        except AttributeError:
            num_classes = model.powerset.num_classes

        overlap_start = num_speakers + 1
        if overlap_start >= num_classes:
            logger.warning(
                "No overlap classes in powerset model "
                "(num_speakers=%d, num_classes=%d).",
                num_speakers, num_classes,
            )
            return []

        indices = list(range(overlap_start, num_classes))
        logger.debug(
            "Powerset: %d overlap classes %s (of %d total, %d speakers)",
            len(indices), indices, num_classes, num_speakers,
        )
        return indices

    except Exception as exc:
        logger.warning(
            "get_overlap_class_indices failed: %s — "
            "concurrency_confidence will not be extracted.",
            exc,
        )
        return []


def extract_posteriors(
    pipeline,
    audio_input: dict,
) -> "SlidingWindowFeature | None":
    """
    Extract raw powerset posterior probabilities for a chunk.

    Calls pipeline._segmentation(audio_input) — a second forward pass
    through the segmentation model only. The full pipeline() call has
    already run for this chunk. This call is fast (~0.1s overhead) and
    is covered by torch.no_grad() in the outer loop.

    Parameters
    ----------
    pipeline
        Loaded pyannote Pipeline from Stage 4 result.
    audio_input : dict
        {"waveform": tensor, "sample_rate": int} for the current chunk.

    Returns
    -------
    SlidingWindowFeature | None
        Shape (num_frames, num_powerset_classes), or None on failure.
        None is non-fatal — concurrency_confidence will be None for all
        segments in this chunk.
    """
    try:
        return pipeline._segmentation(audio_input)
    except Exception as exc:
        logger.warning(
            "extract_posteriors: pipeline._segmentation failed: %s. "
            "concurrency_confidence will be None for this chunk.",
            exc,
        )
        return None


def concurrency_confidence_for_segment(
    posteriors: "SlidingWindowFeature",
    overlap_class_indices: list[int],
    seg_start_local: float,
    seg_end_local: float,
) -> float | None:
    """
    Compute mean overlap class probability across a segment's frames.

    For each frame in the segment, sums the softmax probabilities of
    all overlap classes (indices where 2+ speakers are active).
    Averages per-frame sums to produce a single scalar in [0, 1].

    This is the concurrency_confidence stored on the TimelineSegment.
    A value of 0.94 means the model was highly confident this was
    concurrent speech. A value of 0.51 is a Ghost Speaker — barely
    above the decision boundary.

    Parameters
    ----------
    posteriors : SlidingWindowFeature
        Output of extract_posteriors() — shape (num_frames, num_classes).
    overlap_class_indices : list[int]
        Indices of overlap classes from get_overlap_class_indices().
    seg_start_local, seg_end_local : float
        Segment boundaries in chunk-local time (seconds from chunk start).

    Returns
    -------
    float | None
        Mean per-frame overlap probability in [0, 1], clipped.
        None if the segment has no frames or any operation fails.
    """
    try:
        from pyannote.core import Segment as PySegment
        seg_obj = PySegment(seg_start_local, seg_end_local)

        # mode='loose' includes frames that partially overlap the boundary.
        frames = posteriors.crop(seg_obj, mode="loose")

        if frames is None or len(frames) == 0:
            return None

        # frames: (num_frames_in_segment, num_classes)
        overlap_probs = frames[:, overlap_class_indices]
        per_frame_overlap = overlap_probs.sum(axis=1)
        return float(np.clip(np.mean(per_frame_overlap), 0.0, 1.0))

    except Exception as exc:
        logger.debug(
            "concurrency_confidence_for_segment failed [%.2fs, %.2fs]: %s",
            seg_start_local, seg_end_local, exc,
        )
        return None