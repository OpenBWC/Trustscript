"""
src/utils/audio/models.py
==========================
TrustScript Phase 1 — Shared Audio Data Models

Responsibility
--------------
Dataclasses that represent raw audio measurements across all audio
stages. These are plain data containers with no logic — they exist
so that hashing.py, probe.py, stage1.py, profiling.py, and events.py
can share structured types without circular imports.

Signal Time contract
--------------------
Every measurement that references a position in the audio stream
carries BOTH a sample offset (primary key) and a seconds value
(display format). Sample offsets are computed from the original
sample rate and never accumulate floating point error.

    start_sample: int    ← primary key, never rounded
    start_seconds: float ← derived display value: start_sample / sr

See events.py and profiling.py for the conversion logic.

Stages and their dataclasses
-----------------------------
    Stage 1:  AudioProperties, LoudnessProperties
    Stage 2:  QualityWindow, QualityEvent, AudioQualityProfile, Stage2Result
    Stage 3:  NormalizationResult  (to be added)

Contributors
------------
    Keep this module free of logic. Dataclasses only. When adding new
    stages, add their output dataclasses here and import them from the
    owning stage module.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stage 1 models
# ---------------------------------------------------------------------------


@dataclass
class AudioProperties:
    """
    Raw audio stream properties extracted from an input file via ffprobe.

    Attributes
    ----------
    sample_rate : int
        Audio sample rate in Hz (e.g. 48000, 44100, 16000).
    channels : int
        Number of audio channels (1 = mono, 2 = stereo).
    codec_name : str
        Audio codec as reported by ffprobe.
    duration_seconds : float
        Total duration of the audio stream in seconds.
    bit_rate : int | None
        Audio bit rate in bits per second. None when unavailable.
    """
    sample_rate: int
    channels: int
    codec_name: str
    duration_seconds: float
    bit_rate: int | None = None


@dataclass
class LoudnessProperties:
    """
    Integrated loudness measurement from an ffmpeg loudnorm analysis pass.

    Attributes
    ----------
    integrated_lufs : float
        Integrated loudness in LUFS. Stage 3 target: -16.0 LUFS.
    loudness_range : float
        Loudness Range (LRA) in LU. Informational only.
    true_peak : float
        Maximum true peak level in dBTP. Values >= 0.0 indicate clipping.
    """
    integrated_lufs: float
    loudness_range: float
    true_peak: float


# ---------------------------------------------------------------------------
# Stage 2 models
# ---------------------------------------------------------------------------


@dataclass
class QualityWindow:
    """
    A 30-second quality measurement window with Signal Time anchors.

    QualityWindow represents the audio quality within a fixed-size
    time slice of the recording. Windows are non-overlapping and cover
    the full file duration from t=0 to t=end.

    Signal Time fields
    ------------------
    start_sample and end_sample are expressed in the ORIGINAL file's
    sample space (at original_sample_rate, not the 16kHz analysis rate).
    start_seconds and end_seconds are derived display values:
        start_seconds = start_sample / original_sample_rate

    Attributes
    ----------
    start_sample : int
        First sample of this window in the original recording.
        Primary Signal Time key — never rounded or accumulated.
    end_sample : int
        Last sample (exclusive) of this window in the original recording.
    start_seconds : float
        Display value: start_sample / original_sample_rate.
    end_seconds : float
        Display value: end_sample / original_sample_rate.
    snr_db : float | None
        Local SNR estimate for this window. None if insufficient
        silence frames were available for estimation.
    snr_classification : str
        SNR quality label: EXCELLENT, GOOD, MODERATE, POOR, CRITICAL,
        or UNKNOWN.
    snr_flagged : bool
        True for POOR, CRITICAL, or UNKNOWN classifications.
    noise_stability_cv : float | None
        Coefficient of Variation of silence-frame RMS values within
        this window. None if insufficient silence frames.
    noise_is_unstable : bool
        True when noise_stability_cv > 0.30.
    speech_fraction : float | None
        Fraction of frames classified as speech in this window.
    clipping_detected : bool
        True if any frame in this window exceeded -1 dBFS.
    """
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    snr_db: float | None
    snr_classification: str
    snr_flagged: bool
    noise_stability_cv: float | None
    noise_is_unstable: bool
    speech_fraction: float | None
    clipping_detected: bool

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "snr_db": round(self.snr_db, 2) if self.snr_db is not None else None,
            "snr_classification": self.snr_classification,
            "snr_flagged": self.snr_flagged,
            "noise_stability_cv": (
                round(self.noise_stability_cv, 4)
                if self.noise_stability_cv is not None else None
            ),
            "noise_is_unstable": self.noise_is_unstable,
            "speech_fraction": (
                round(self.speech_fraction, 3)
                if self.speech_fraction is not None else None
            ),
            "clipping_detected": self.clipping_detected,
        }


@dataclass
class QualityEvent:
    """
    A contiguous zone of degraded audio quality with Signal Time anchors.

    QualityEvents are produced by merging adjacent flagged sliding windows
    into intervals. They identify WHERE in the recording a quality problem
    occurs, not just WHETHER the file has a problem.

    Event types
    -----------
    LOW_SNR_ZONE        SNR below POOR threshold (< 10 dB) sustained
                        across multiple consecutive windows.
    UNSTABLE_NOISE_ZONE Non-stationary noise floor (CV > 0.30) sustained
                        across multiple consecutive windows.
    CLIPPING_ZONE       Sustained clipping detected across frames.
    COMBINED            Multiple quality problems co-occurring in the
                        same time region.

    Signal Time fields
    ------------------
    Same contract as QualityWindow: start_sample and end_sample are in
    the original sample space. Seconds values are derived display values.

    Attributes
    ----------
    event_type : str
        One of: LOW_SNR_ZONE, UNSTABLE_NOISE_ZONE, CLIPPING_ZONE, COMBINED.
    start_sample : int
        First sample of the degraded zone in the original recording.
    end_sample : int
        Last sample (exclusive) of the degraded zone.
    start_seconds : float
        Display value: start_sample / original_sample_rate.
    end_seconds : float
        Display value: end_sample / original_sample_rate.
    duration_seconds : float
        Duration of the event: end_seconds - start_seconds.
    snr_db : float | None
        Mean SNR across sliding windows in this event zone.
    snr_classification : str | None
        SNR classification for the event zone.
    noise_stability_cv : float | None
        Mean CV across sliding windows in this event zone.
    noise_is_unstable : bool
        True if the majority of windows in this zone were unstable.
    note : str
        Human-readable description of the event and its forensic
        implications for diarization reliability.
    """
    event_type: str
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    snr_db: float | None
    snr_classification: str | None
    noise_stability_cv: float | None
    noise_is_unstable: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "snr_db": round(self.snr_db, 2) if self.snr_db is not None else None,
            "snr_classification": self.snr_classification,
            "noise_stability_cv": (
                round(self.noise_stability_cv, 4)
                if self.noise_stability_cv is not None else None
            ),
            "noise_is_unstable": self.noise_is_unstable,
            "note": self.note,
        }


@dataclass
class AudioQualityProfile:
    """
    Complete audio quality characterization from Stage 2.

    Structured as three layers:
        file_level   — aggregate summary across the full recording
        quality_windows — 30-second non-overlapping quality slices
        quality_events  — contiguous zones of degraded quality

    All time references use sample offsets as primary keys.
    Seconds values are derived display values only.

    Attributes
    ----------
    duration_seconds : float
        Total audio duration from Stage 1 ffprobe.
    sample_rate_original : int
        Original sample rate from Stage 1 ffprobe.
    channels_original : int
        Original channel count from Stage 1 ffprobe.
    clipping_detected : bool
        True if any frame in the full file exceeded -1 dBFS.
    clipping_peak_dbfs : float | None
        Maximum peak amplitude in dBFS across the full file.
    snr_db : float | None
        File-level SNR estimate (mean of per-window SNRs).
    snr_classification : str
        File-level SNR quality label.
    snr_flagged : bool
        True for POOR, CRITICAL, or UNKNOWN file-level SNR.
    noise_stability_cv : float | None
        File-level CV of silence-frame RMS values.
    noise_is_unstable : bool
        True when file-level noise_stability_cv > 0.30.
    vad_speech_fraction : float | None
        Fraction of all frames classified as speech.
    vad_energy_percentiles : dict[str, float]
        p10/p20/p50/p80/p90 RMS energy values across the full file.
    quality_windows : list[QualityWindow]
        30-second quality measurement windows with Signal Time anchors.
    quality_events : list[QualityEvent]
        Contiguous zones of degraded audio quality with Signal Time anchors.
    snr_note : str | None
        Caveat about the SNR measurement, if applicable.
    """
    duration_seconds: float
    sample_rate_original: int
    channels_original: int
    clipping_detected: bool
    clipping_peak_dbfs: float | None
    snr_db: float | None
    snr_classification: str
    snr_flagged: bool
    noise_stability_cv: float | None
    noise_is_unstable: bool
    vad_speech_fraction: float | None
    quality_windows: list[QualityWindow] = field(default_factory=list)
    quality_events: list[QualityEvent] = field(default_factory=list)
    vad_energy_percentiles: dict[str, float] = field(default_factory=dict)
    snr_note: str | None = None

    def to_dict(self) -> dict:
        """
        Serialize to the `audio_quality` block in phase1.json.

        Three-layer structure:
            file_level       — aggregate summary
            quality_windows  — 30s time-indexed slices
            quality_events   — contiguous degraded zones
        """
        return {
            "file_level": {
                "snr_db": (
                    round(self.snr_db, 2) if self.snr_db is not None else None
                ),
                "snr_classification": self.snr_classification,
                "snr_flagged": self.snr_flagged,
                "noise_stability_cv": (
                    round(self.noise_stability_cv, 4)
                    if self.noise_stability_cv is not None else None
                ),
                "noise_is_unstable": self.noise_is_unstable,
                "duration_seconds": round(self.duration_seconds, 3),
                "sample_rate_original": self.sample_rate_original,
                "channels_original": self.channels_original,
                "clipping_detected": self.clipping_detected,
                "clipping_peak_dbfs": (
                    round(self.clipping_peak_dbfs, 2)
                    if self.clipping_peak_dbfs is not None else None
                ),
                "vad_speech_fraction": (
                    round(self.vad_speech_fraction, 3)
                    if self.vad_speech_fraction is not None else None
                ),
                "vad_energy_percentiles": self.vad_energy_percentiles or None,
                "snr_note": self.snr_note,
            },
            "quality_windows": [w.to_dict() for w in self.quality_windows],
            "quality_events": [e.to_dict() for e in self.quality_events],
        }


@dataclass
class Stage2Result:
    """
    Complete output of Stage 2: Audio Quality Profiling.

    Attributes
    ----------
    audio_quality : AudioQualityProfile
        All quality measurements from the Stage 2 profiling pass.
    """
    audio_quality: AudioQualityProfile

    def to_processing_notes(self) -> dict:
        """
        Serialize Stage 2 results into the processing_notes dict format.
        Called by Stage 7 (Output Assembly, src/utils/schema.py).
        """
        return {
            "stage_2": {
                "vad_method": "energy_based",
                "sliding_window_seconds": 5.0,
                "sliding_hop_seconds": 1.0,
                "quality_window_seconds": 30.0,
                "signal_time_primary_key": "sample_offset",
                "vad_note": (
                    "SNR estimated via energy-based frame-level VAD. "
                    "No neural model required. Signal Time anchored to "
                    "original sample rate — seconds values are derived "
                    "display values only. Refined SNR via pyannote VAD "
                    "produced in Stage 5."
                ),
                "audio_quality": self.audio_quality.to_dict(),
            }
        }