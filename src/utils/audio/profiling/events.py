"""
src/utils/audio/events.py
==========================
TrustScript Phase 1 — Stage 2: Quality Event Detection

Responsibility
--------------
Convert per-window quality measurements into two structured outputs:

    1. QualityWindows  — 30-second non-overlapping quality slices
                         covering the full file with Signal Time anchors.

    2. QualityEvents   — Contiguous zones of degraded audio quality,
                         produced by merging adjacent flagged sliding
                         windows into intervals.

This module does no audio I/O. It operates entirely on arrays of
per-frame measurements produced by profiling/ streaming pass.

Signal Time contract
--------------------
All time values are derived from sample offsets in the ORIGINAL file's
sample space. The conversion:

    original_start_sample = round(
        analysis_frame_index * FRAME_SAMPLES * (original_sr / analysis_sr)
    )
    start_seconds = original_start_sample / original_sr

This is computed once per frame and never re-accumulated, preventing
IEEE 754 floating point error from compounding across long files.

The "movie review vs. timestamps" principle
--------------------------------------------
File-level summary statistics tell you whether a recording is "Good"
or "Bad." QualityEvents tell you the timestamp of the jump scares.

A 2-hour file with an average SNR of 25 dB (EXCELLENT) may contain
a 30-second siren burst at t=912s that the summary hides. QualityEvents
surface that event precisely so Phase 4 Fusion and GroundTruth reviewers
know exactly where to look.

Contributors
------------
    _QUALITY_WINDOW_SECONDS and the event merging gap tolerance
    (_EVENT_MERGE_GAP_SECONDS) are tunable. After collecting a real
    BWC corpus, calibrate these against observed event patterns.
    Document the calibration dataset in a comment when you do.
"""

import logging
from dataclasses import dataclass

import numpy as np

from .constants import (
    EPSILON,
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    MIN_SILENCE_FRAMES,
    NOISE_INSTABILITY_THRESHOLD,
    QUALITY_WINDOW_SECONDS,
    SNR_THRESHOLDS,
    SILENCE_PERCENTILE,
    SPEECH_PERCENTILE,
    CLIPPING_THRESHOLD_LINEAR,
)
from .models import QualityEvent, QualityWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Duration of each non-overlapping quality window in seconds.
_QUALITY_WINDOW_SECONDS: float = 30.0

#: SNR below this value (dB) triggers a LOW_SNR_ZONE event.
#: Matches the POOR/CRITICAL boundary in profiling._SNR_THRESHOLDS.
_EVENT_LOW_SNR_THRESHOLD_DB: float = 10.0

#: CV above this value triggers an UNSTABLE_NOISE_ZONE event.
#: Matches _NOISE_INSTABILITY_THRESHOLD in profiling.py.
_EVENT_INSTABILITY_THRESHOLD: float = 0.30

#: Adjacent flagged sliding windows within this many seconds of each
#: other are merged into a single event. Prevents fragmentation from
#: brief recoveries between related noise bursts.
_EVENT_MERGE_GAP_SECONDS: float = 2.0

#: Minimum event duration in seconds. Events shorter than this are
#: discarded as transient noise rather than sustained quality problems.
_EVENT_MIN_DURATION_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# Internal window data structure (intermediate, not exported)
# ---------------------------------------------------------------------------


@dataclass
class _SlidingWindowResult:
    """
    Quality measurements for one position of the sliding window.
    Internal only — converted to QualityWindow and QualityEvent at output.

    Attributes
    ----------
    start_frame : int
        First frame index in the analysis (16kHz) frame space.
    end_frame : int
        Last frame index (exclusive).
    start_sample : int
        First sample in the ORIGINAL file's sample space.
    end_sample : int
        Last sample (exclusive) in the original sample space.
    start_seconds : float
        Derived display value: start_sample / original_sample_rate.
    end_seconds : float
        Derived display value: end_sample / original_sample_rate.
    snr_db : float | None
    noise_cv : float | None
    speech_fraction : float
    clipping_detected : bool
    """
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    snr_db: float | None
    noise_cv: float | None
    speech_fraction: float
    clipping_detected: bool


# ---------------------------------------------------------------------------
# EventDetector
# ---------------------------------------------------------------------------


class EventDetector:
    """
    Converts per-frame quality measurements into time-indexed windows
    and contiguous quality event intervals.

    Usage
    -----
    Instantiate with the original sample rate and analysis parameters,
    then call build_quality_windows() and detect_quality_events().

        detector = EventDetector(
            original_sample_rate=48000,
            analysis_sample_rate=16000,
            frame_samples=320,
        )
        windows = detector.build_quality_windows(sliding_results)
        events  = detector.detect_quality_events(sliding_results)

    Parameters
    ----------
    original_sample_rate : int
        Sample rate of the original file (e.g. 48000).
    analysis_sample_rate : int
        Sample rate used during profiling analysis (always 16000).
    frame_samples : int
        Number of samples per analysis frame at analysis_sample_rate.
        For 20ms frames at 16kHz: 320.
    """

    def __init__(
        self,
        original_sample_rate: int,
        analysis_sample_rate: int,
        frame_samples: int,
    ) -> None:
        self.original_sr = original_sample_rate
        self.analysis_sr = analysis_sample_rate
        self.frame_samples = frame_samples

        # Scale factor: converts analysis frame index → original sample offset.
        # Computed once and reused — never accumulated.
        self._analysis_to_original_scale = original_sample_rate / analysis_sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def frame_to_original_sample(self, frame_index: int) -> int:
        """
        Convert an analysis frame index to a sample offset in the
        original file's sample space.

        This is the core Signal Time computation. It is called once per
        frame and uses integer arithmetic to prevent floating point
        accumulation error.

        Parameters
        ----------
        frame_index : int
            0-based frame index in the analysis (16kHz) stream.

        Returns
        -------
        int
            Sample offset in the original recording. Use this as the
            primary Signal Time key. Convert to seconds for display only:
                seconds = sample / original_sample_rate
        """
        analysis_sample = frame_index * self.frame_samples
        return round(analysis_sample * self._analysis_to_original_scale)

    def frames_to_seconds(self, frame_index: int) -> float:
        """
        Convert a frame index to display seconds in the original timeline.

        For internal logging and display only. Never use the return value
        as a Signal Time key — always use frame_to_original_sample() for
        that purpose.
        """
        return self.frame_to_original_sample(frame_index) / self.original_sr

    def build_quality_windows(
        self,
        rms_per_frame: np.ndarray,
        peak_per_frame: np.ndarray,
        silence_threshold: float,
        speech_threshold: float,
        noise_cv_per_frame: np.ndarray | None = None,
    ) -> list[QualityWindow]:
        """
        Aggregate per-frame measurements into 30-second QualityWindows.

        Windows are non-overlapping and cover the full file from frame 0
        to the last frame. Each window carries Signal Time anchors
        (start_sample, end_sample) in the original sample space.

        Parameters
        ----------
        rms_per_frame : np.ndarray
            Per-frame RMS values from the streaming pass.
        peak_per_frame : np.ndarray
            Per-frame peak amplitude values from the streaming pass.
        silence_threshold : float
            Global silence RMS threshold from file-level VAD.
        speech_threshold : float
            Global speech RMS threshold from file-level VAD.
        noise_cv_per_frame : np.ndarray | None
            Pre-computed per-frame CV values, if available.
            If None, CV is computed per window from the silence frames.

        Returns
        -------
        list[QualityWindow]
            30-second windows covering the full file, ordered by
            start_sample ascending.
        """
        # Import here to avoid circular dependency at module load time.
        frames_per_window = int(
            QUALITY_WINDOW_SECONDS * 1000 / FRAME_DURATION_MS
        )
        n_frames = len(rms_per_frame)
        windows: list[QualityWindow] = []

        for window_start in range(0, n_frames, frames_per_window):
            window_end = min(window_start + frames_per_window, n_frames)
            window_rms = rms_per_frame[window_start:window_end]
            window_peak = peak_per_frame[window_start:window_end]

            # Signal Time anchors for this window.
            start_sample = self.frame_to_original_sample(window_start)
            end_sample = self.frame_to_original_sample(window_end)
            start_seconds = start_sample / self.original_sr
            end_seconds = end_sample / self.original_sr

            # Local speech/silence classification using global thresholds.
            silence_frames = window_rms[window_rms <= silence_threshold]
            speech_frames = window_rms[window_rms >= speech_threshold]
            speech_fraction = float(len(speech_frames)) / len(window_rms)

            # Local SNR.
            snr_db: float | None = None
            if (
                len(silence_frames) >= MIN_SILENCE_FRAMES
                and len(speech_frames) > 0
            ):
                mean_speech = float(np.mean(speech_frames))
                mean_silence = float(np.mean(silence_frames))
                if mean_silence > EPSILON:
                    snr_db = float(
                        20.0 * np.log10(mean_speech / mean_silence)
                    )

            # Local SNR classification.
            snr_classification = "UNKNOWN"
            snr_flagged = True
            if snr_db is not None:
                for min_snr, label, flagged in SNR_THRESHOLDS:
                    if snr_db >= min_snr:
                        snr_classification = label
                        snr_flagged = flagged
                        break

            # Local noise stability (CV) from silence frames in this window.
            noise_cv: float | None = None
            noise_is_unstable = False
            if len(silence_frames) >= MIN_SILENCE_FRAMES:
                mean_sil = float(np.mean(silence_frames))
                std_sil = float(np.std(silence_frames))
                noise_cv = std_sil / (mean_sil + EPSILON)
                noise_is_unstable = noise_cv > NOISE_INSTABILITY_THRESHOLD

            # Clipping: any frame peak in this window above threshold.
            clipping_detected = bool(np.any(window_peak >= CLIPPING_THRESHOLD_LINEAR))

            windows.append(QualityWindow(
                start_sample=start_sample,
                end_sample=end_sample,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                snr_db=snr_db,
                snr_classification=snr_classification,
                snr_flagged=snr_flagged,
                noise_stability_cv=noise_cv,
                noise_is_unstable=noise_is_unstable,
                speech_fraction=speech_fraction,
                clipping_detected=clipping_detected,
            ))

        logger.debug(
            "Built %d quality windows (%.0f seconds each)",
            len(windows),
            _QUALITY_WINDOW_SECONDS,
        )
        return windows

    def detect_quality_events(
        self,
        sliding_results: list[_SlidingWindowResult],
    ) -> list[QualityEvent]:
        """
        Detect contiguous zones of degraded audio quality.

        Merges adjacent flagged sliding windows into event intervals.
        Adjacent windows within _EVENT_MERGE_GAP_SECONDS of each other
        are merged to prevent fragmentation from brief recoveries.

        Event types are assigned based on which quality dimensions are
        degraded in the merged interval:
            LOW_SNR_ZONE        SNR < _EVENT_LOW_SNR_THRESHOLD_DB
            UNSTABLE_NOISE_ZONE CV > _EVENT_INSTABILITY_THRESHOLD
            CLIPPING_ZONE       peak >= -1 dBFS
            COMBINED            multiple quality problems co-occurring

        Parameters
        ----------
        sliding_results : list[_SlidingWindowResult]
            Per-sliding-window measurements from profiling.py.

        Returns
        -------
        list[QualityEvent]
            Contiguous degraded zones ordered by start_sample ascending.
            Events shorter than _EVENT_MIN_DURATION_SECONDS are discarded.
        """
        if not sliding_results:
            return []

        # Flag each sliding window for each problem type.
        flagged: list[dict] = []
        for w in sliding_results:
            low_snr = (
                w.snr_db is not None
                and w.snr_db < _EVENT_LOW_SNR_THRESHOLD_DB
            )
            unstable = (
                w.noise_cv is not None
                and w.noise_cv > _EVENT_INSTABILITY_THRESHOLD
            )
            clipping = w.clipping_detected

            if low_snr or unstable or clipping:
                flagged.append({
                    "window": w,
                    "low_snr": low_snr,
                    "unstable": unstable,
                    "clipping": clipping,
                })

        if not flagged:
            logger.debug("No quality events detected.")
            return []

        # Merge adjacent flagged windows.
        merge_gap_samples = int(
            _EVENT_MERGE_GAP_SECONDS * self.original_sr
        )
        merged_groups: list[list[dict]] = []
        current_group: list[dict] = [flagged[0]]

        for item in flagged[1:]:
            prev_end = current_group[-1]["window"].end_sample
            curr_start = item["window"].start_sample
            if curr_start - prev_end <= merge_gap_samples:
                current_group.append(item)
            else:
                merged_groups.append(current_group)
                current_group = [item]
        merged_groups.append(current_group)

        # Convert merged groups to QualityEvent objects.
        events: list[QualityEvent] = []
        min_samples = int(_EVENT_MIN_DURATION_SECONDS * self.original_sr)

        for group in merged_groups:
            start_sample = group[0]["window"].start_sample
            end_sample = group[-1]["window"].end_sample
            duration_samples = end_sample - start_sample

            # Discard events shorter than the minimum duration.
            if duration_samples < min_samples:
                continue

            start_seconds = start_sample / self.original_sr
            end_seconds = end_sample / self.original_sr
            duration_seconds = end_seconds - start_seconds

            # Aggregate metrics across the group.
            snr_values = [
                g["window"].snr_db for g in group
                if g["window"].snr_db is not None
            ]
            cv_values = [
                g["window"].noise_cv for g in group
                if g["window"].noise_cv is not None
            ]
            mean_snr = float(np.mean(snr_values)) if snr_values else None
            mean_cv = float(np.mean(cv_values)) if cv_values else None

            any_low_snr = any(g["low_snr"] for g in group)
            any_unstable = any(g["unstable"] for g in group)
            any_clipping = any(g["clipping"] for g in group)

            # Determine event type and human-readable note.
            active_types = sum([any_low_snr, any_unstable, any_clipping])
            if active_types > 1:
                event_type = "COMBINED"
                note = self._build_combined_note(
                    any_low_snr, any_unstable, any_clipping,
                    mean_snr, mean_cv, duration_seconds,
                )
            elif any_low_snr:
                event_type = "LOW_SNR_ZONE"
                note = (
                    f"Sustained low-SNR zone ({duration_seconds:.1f}s). "
                    f"Mean SNR: {mean_snr:.1f} dB. "
                    "Possible cause: siren, wind burst, or physical altercation. "
                    "Diarization reliability is reduced in this region."
                )
            elif any_unstable:
                event_type = "UNSTABLE_NOISE_ZONE"
                note = (
                    f"Non-stationary noise floor ({duration_seconds:.1f}s). "
                    f"Mean CV: {mean_cv:.3f}. "
                    "Possible cause: rain, wind gusts, or intermittent radio. "
                    "SNR estimates in this region may be over-optimistic."
                )
            else:
                event_type = "CLIPPING_ZONE"
                note = (
                    f"Sustained clipping detected ({duration_seconds:.1f}s). "
                    "The recording was captured at too high a level. "
                    "Clipping cannot be recovered through normalization."
                )

            # SNR classification for this event.
            snr_classification: str | None = None
            if mean_snr is not None:
                for min_snr, label, _ in SNR_THRESHOLDS:
                    if mean_snr >= min_snr:
                        snr_classification = label
                        break

            events.append(QualityEvent(
                event_type=event_type,
                start_sample=start_sample,
                end_sample=end_sample,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=duration_seconds,
                snr_db=round(mean_snr, 2) if mean_snr is not None else None,
                snr_classification=snr_classification,
                noise_stability_cv=(
                    round(mean_cv, 4) if mean_cv is not None else None
                ),
                noise_is_unstable=any_unstable,
                note=note,
            ))

        logger.info(
            "Detected %d quality event(s) across %.1f seconds of "
            "degraded audio.",
            len(events),
            sum(e.duration_seconds for e in events),
        )

        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_combined_note(
        low_snr: bool,
        unstable: bool,
        clipping: bool,
        mean_snr: float | None,
        mean_cv: float | None,
        duration: float,
    ) -> str:
        """Build a human-readable note for COMBINED quality events."""
        parts = []
        if low_snr and mean_snr is not None:
            parts.append(f"low SNR ({mean_snr:.1f} dB)")
        elif low_snr:
            parts.append("low SNR (unknown)")
        if unstable and mean_cv is not None:
            parts.append(f"unstable noise floor (CV={mean_cv:.3f})")
        elif unstable:
            parts.append("unstable noise floor")
        if clipping:
            parts.append("clipping")

        problems = ", ".join(parts)
        return (
            f"Multiple quality problems co-occurring ({duration:.1f}s): "
            f"{problems}. "
            "This region has severely reduced diarization reliability. "
            "Phase 4 Fusion should weight all segments in this zone down."
        )