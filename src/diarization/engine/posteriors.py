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
from pyannote.core import Segment as PySegment

if TYPE_CHECKING:
    from pyannote.core import SlidingWindowFeature

logger = logging.getLogger(__name__)


def get_overlap_class_indices(pipeline) -> list[int]:
    """
    Identify which class indices represent concurrent speech.

    For pyannote/speaker-diarization-community-1, the segmentation model
    returns a 3D posterior tensor of shape (windows, frames_per_window, 3)
    where the 3 classes are: [silence, speech, overlap].
    Class index 2 is the overlap class.

    This is different from the full powerset format used in older pyannote
    versions where overlap classes were at indices N+1..end. The community-1
    model uses a simpler 3-class segmentation — the `powerset` attribute
    does not exist on PyanNet, which is why the original introspection fails.

    Returns [2] for the community-1 model. Returns [] on any failure —
    callers degrade gracefully by setting concurrency_confidence = None.
    """
    try:
        # Verify the pipeline has a _segmentation attribute at all.
        if not hasattr(pipeline, "_segmentation"):
            logger.warning(
                "get_overlap_class_indices: pipeline has no _segmentation "
                "attribute — concurrency_confidence will not be extracted."
            )
            return []

        # For community-1 (PyanNet), the segmentation output is
        # (windows, frames_per_window, 3) where index 2 = overlap.
        # Return immediately without attempting powerset introspection.
        logger.debug(
            "get_overlap_class_indices: using community-1 3-class format "
            "(silence=0, speech=1, overlap=2)."
        )
        return [2]

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
        None if the segment has no frames, the internal API is
        unavailable, or any operation fails.

        Fallback contract: downstream phases (Phase 2 triage,
        stage5.py flag assignment) must treat None as UNKNOWN —
        neither safe (0.0) nor confirmed (1.0). Concurrency-specific
        confidence weighting must be bypassed for None segments.
        Do not substitute a default float; preserve None so the
        absence of data remains distinguishable from a low score.
    """
    try:
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