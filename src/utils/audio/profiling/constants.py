"""
src/utils/audio/constants.py
=============================
TrustScript Phase 1 — Stage 2 Shared Constants

Responsibility
--------------
All tunable parameters for Stage 2 audio quality profiling. Centralised
here so that streaming.py, vad.py, snr.py, events.py, and stage2.py all
import from a single source of truth rather than duplicating values.

Calibration note
----------------
The VAD percentile thresholds (_SILENCE_PERCENTILE, _SPEECH_PERCENTILE),
noise instability threshold (_NOISE_INSTABILITY_THRESHOLD), and SNR
classification thresholds (_SNR_THRESHOLDS) are starting hypotheses.

After collecting a real BWC corpus, calibrate these values by plotting:
    - SNR distributions vs. reid_score degradation
    - noise_stability_cv distributions vs. diarization error rate
    - event duration distributions vs. human review outcomes

Document the calibration dataset and methodology in a comment here
when you update these values.

Contributors
------------
    Do not add logic to this module. Constants only.
    If a value is only used in one module, keep it there.
    If a value is shared across two or more modules, it belongs here.
"""

# ---------------------------------------------------------------------------
# Analysis frame parameters
# ---------------------------------------------------------------------------

#: Sample rate used for the profiling analysis pass (Hz).
#: The original sample rate is preserved from Stage 1 and written to
#: output separately — this is only used for analysis computation.
#: 16kHz is sufficient for speech energy analysis and minimises memory use.
ANALYSIS_SAMPLE_RATE: int = 16_000

#: Duration of each analysis frame in milliseconds.
#: 20ms is standard for speech processing — long enough for stable
#: energy estimates, short enough to capture rapid speech transitions.
FRAME_DURATION_MS: float = 20.0

#: Number of samples per analysis frame at ANALYSIS_SAMPLE_RATE.
#: 16000 Hz × 0.020 s = 320 samples.
FRAME_SAMPLES: int = int(ANALYSIS_SAMPLE_RATE * FRAME_DURATION_MS / 1000.0)

#: Small epsilon to prevent log(0) and division-by-zero in dB/RMS math.
EPSILON: float = 1e-10

#: Linear amplitude equivalent of the -1 dBFS clipping threshold.
#: 10^(-1/20) ≈ 0.8913. Samples with abs(amplitude) ≥ this value
#: are at or above the clipping boundary.
CLIPPING_THRESHOLD_LINEAR: float = 10 ** (-1.0 / 20.0)


# ---------------------------------------------------------------------------
# Energy-based VAD parameters
# ---------------------------------------------------------------------------

#: Frames at or below this RMS energy percentile are classified as silence.
#: 20th percentile captures quiet background noise in typical BWC audio.
SILENCE_PERCENTILE: float = 20.0

#: Frames at or above this RMS energy percentile are classified as speech.
#: Frames between SILENCE_PERCENTILE and SPEECH_PERCENTILE are ambiguous
#: and excluded from SNR computation to avoid contaminating either estimate.
SPEECH_PERCENTILE: float = 50.0

#: Minimum number of silence frames required for a reliable noise floor
#: estimate. Below this count, std() is statistically meaningless and
#: CV-based stationarity checks produce noise rather than signal.
MIN_SILENCE_FRAMES: int = 10

#: Minimum number of frames required before VAD percentile computation
#: is considered statistically meaningful. A 30-second quality window
#: at the tail of a file may contain only 2 seconds of audio — below
#: this threshold, percentile-based thresholds are unreliable and the
#: window is skipped rather than producing a misleading SNR estimate.
#: Similarly, a 5-second sliding window sliver at EOF is discarded.
#: Expressed in frames: 2s / 20ms = 100 frames.
MIN_WINDOW_FRAMES: int = 100  # 2 seconds at 20ms per frame


# ---------------------------------------------------------------------------
# Noise stationarity
# ---------------------------------------------------------------------------

#: Coefficient of Variation (CV) above this value indicates a
#: non-stationary noise floor. The "silence" frames show too much
#: energy variation to represent a stable background — the SNR
#: estimate may be over-optimistic.
#:
#: CV interpretation:
#:   < 0.15   Stable, stationary (air conditioning, engine idle)
#:   0.15–0.30 Moderate variation, SNR is reasonable
#:   > 0.30   Unstable (rain, wind gusts, distant shouting)
NOISE_INSTABILITY_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Sliding window parameters
# ---------------------------------------------------------------------------

#: Duration of the sliding window used for per-window SNR computation (s).
#: 5 seconds gives sufficient frames (250 at 20ms) for stable SNR estimates
#: while being short enough to detect transient quality events.
SLIDING_WINDOW_SECONDS: float = 5.0

#: Hop between consecutive sliding window positions (s).
#: 1-second hop gives 1Hz temporal resolution on the SNR time series.
SLIDING_HOP_SECONDS: float = 1.0

#: Duration of each non-overlapping quality window written to output (s).
#: 30 seconds provides 1,500 frames per window — sufficient statistical
#: density while keeping the quality_windows array to a manageable size
#: (~120 entries for a 2-hour file).
QUALITY_WINDOW_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# SNR classification thresholds
# ---------------------------------------------------------------------------

#: Ordered list of (min_snr_db_inclusive, classification_label, snr_flagged).
#: Evaluated top-down — the first entry whose threshold the measured SNR
#: meets or exceeds is used.
#:
#: snr_flagged=True for POOR and CRITICAL: these conditions reduce
#: diarization reliability and Phase 4 Fusion should weight segments
#: in these zones down.
#:
#: Reference thresholds (starting hypotheses — calibrate against BWC corpus):
#:   > 30 dB  EXCELLENT   Clean indoor, close mic
#:   20–30 dB GOOD        Normal outdoor, light wind
#:   15–20 dB MODERATE    Active scene, siren/radio
#:   10–15 dB POOR        Heavy wind, multiple sirens
#:   < 10 dB  CRITICAL    Near-unusable noise floor
SNR_THRESHOLDS: list[tuple[float, str, bool]] = [
    (30.0,          "EXCELLENT", False),
    (20.0,          "GOOD",      False),
    (15.0,          "MODERATE",  False),
    (10.0,          "POOR",      True),
    (float("-inf"), "CRITICAL",  True),
]