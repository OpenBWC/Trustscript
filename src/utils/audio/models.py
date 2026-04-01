"""
src/utils/audio/models.py
==========================
TrustScript Phase 1 — Shared Audio Data Models

Responsibility
--------------
Dataclasses that represent raw audio measurements across all audio
stages. These are plain data containers with no logic — they exist
so that hashing.py, probe.py, stage1.py, and profiling.py can share
structured types without any of those modules importing from each other.

Stages and their dataclasses
-----------------------------
    Stage 1:  AudioProperties, LoudnessProperties  (probed from source file)
    Stage 2:  AudioQualityProfile, Stage2Result     (computed from waveform)
    Stage 3:  NormalizationResult                   (to be added)

Contributors
------------
    Keep this module free of logic. Dataclasses only. Any function
    that computes, transforms, or validates belongs in the module
    that owns that stage. When adding new stages, add their output
    dataclasses here.
"""


from dataclasses import dataclass


@dataclass
class AudioProperties:
    """
    Raw audio stream properties extracted from an input file via ffprobe.

    These reflect the file's actual state before any normalization.
    All Stage 1 decision logic reads from this dataclass rather than
    re-probing the file.

    Attributes
    ----------
    sample_rate : int
        Audio sample rate in Hz (e.g. 48000, 44100, 16000).
    channels : int
        Number of audio channels (1 = mono, 2 = stereo).
    codec_name : str
        Audio codec as reported by ffprobe (e.g. "pcm_s16le", "aac", "mp3").
    duration_seconds : float
        Total duration of the audio stream in seconds.
    bit_rate : int | None
        Audio bit rate in bits per second. None when ffprobe cannot
        determine bit rate without a full decode (e.g. some lossless formats).
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

    Produced by probe.measure_loudness(). Requires a full decode of the
    audio content, so it is only computed when needed for the passthrough
    decision (see stage1.run_stage1() for the conditional logic).

    Attributes
    ----------
    integrated_lufs : float
        Integrated loudness in LUFS (Loudness Units Full Scale).
        TrustScript's Stage 3 target is -16.0 LUFS per EBU R128.
    loudness_range : float
        Loudness Range (LRA) in LU. High LRA indicates highly dynamic
        audio (e.g. quiet speech followed by shouting). Informational only —
        not used in passthrough decisions but written to output metadata.
    true_peak : float
        Maximum true peak level in dBTP (decibels True Peak).
        Values above 0.0 indicate digital clipping. Used as a secondary
        clipping indicator in Stage 2 Audio Quality Profiling.
    """
    integrated_lufs: float
    loudness_range: float
    true_peak: float
    

# ---------------------------------------------------------------------------
# Stage 2 models
# ---------------------------------------------------------------------------


@dataclass
class AudioQualityProfile:
    """
    Audio quality measurements from Stage 2: Audio Quality Profiling.

    All measurements are taken from the original input file before
    any normalization. They reflect true recording conditions, not
    the processed output.

    This dataclass is serialized into the `audio_quality` block in
    the final {incident_id}_phase1.json output.

    Attributes
    ----------
    duration_seconds : float
        Total audio duration in seconds. Taken from Stage 1 ffprobe output.
    sample_rate_original : int
        Original sample rate in Hz. Taken from Stage 1 ffprobe output.
    channels_original : int
        Original channel count. Taken from Stage 1 ffprobe output.
    clipping_detected : bool
        True if peak amplitude exceeds -1 dBFS in any part of the file.
        Digital clipping indicates the audio was recorded at too high a
        level and cannot be recovered through normalization.
    clipping_peak_dbfs : float | None
        Actual peak amplitude in dBFS across the full file.
        Negative values indicate headroom below clipping.
        Values >= -1.0 dBFS indicate clipping.
        None if waveform loading failed.
    snr_db : float | None
        Estimated Signal-to-Noise Ratio in decibels. Computed by comparing
        RMS energy of speech-active frames vs. silence frames via an
        energy-based VAD pass. None if SNR could not be estimated (e.g.
        audio is entirely speech or entirely silence).
    snr_classification : str
        Human-readable SNR quality label derived from snr_db.
        One of: EXCELLENT, GOOD, MODERATE, POOR, CRITICAL, UNKNOWN.
    snr_flagged : bool
        True when snr_classification is POOR or CRITICAL (SNR < 15 dB).
        Flagged files have elevated AMBIGUOUS assignment rates and reduced
        diarization reliability. Phase 4 Fusion weights these segments down.
    vad_speech_fraction : float | None
        Fraction of audio frames classified as speech by the energy-based
        VAD (0.0 = all silence, 1.0 = all speech). Informational — not
        used in downstream decisions. None if VAD pass failed.
    snr_note : str | None
        Caveat or explanation about the SNR measurement, if applicable.
        E.g. "Insufficient silence frames for reliable SNR estimate."
        None when the measurement is straightforward.
    """
    duration_seconds: float
    sample_rate_original: int
    channels_original: int
    clipping_detected: bool
    clipping_peak_dbfs: float | None
    snr_db: float | None
    snr_classification: str
    snr_flagged: bool
    vad_speech_fraction: float | None
    snr_note: str | None = None

    def to_dict(self) -> dict:
        """
        Serialize to the `audio_quality` block format expected by the
        TrustScript output JSON schema.

        Returns
        -------
        dict
            Dictionary matching the audio_quality block in phase1.json.
        """
        return {
            "snr_db": round(self.snr_db, 2) if self.snr_db is not None else None,
            "snr_classification": self.snr_classification,
            "snr_flagged": self.snr_flagged,
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
            "snr_note": self.snr_note,
        }


@dataclass
class Stage2Result:
    """
    Complete output of Stage 2: Audio Quality Profiling.

    Passed to Stage 3 (Normalization) and Stage 7 (Output Assembly).
    The audio_quality field is serialized directly into the combined
    output JSON.

    Attributes
    ----------
    audio_quality : AudioQualityProfile
        All quality measurements from the Stage 2 profiling pass.
    """
    audio_quality: AudioQualityProfile

    def to_processing_notes(self) -> dict:
        """
        Serialize Stage 2 results into the processing_notes format
        expected by the TrustScript output JSON schema.

        Called by Stage 7 (Output Assembly, src/utils/schema.py).

        Returns
        -------
        dict
            Stage 2 metadata for inclusion in processing_notes.
        """
        return {
            "stage_2": {
                "vad_method": "energy_based",
                "vad_note": (
                    "SNR estimated via energy-based frame-level VAD. "
                    "No neural model required. Independent of HuggingFace "
                    "authentication. Suitable for pre-normalization quality "
                    "characterization."
                ),
                "audio_quality": self.audio_quality.to_dict(),
            }
        }