"""
src/utils/normalization/constants.py
======================================
TrustScript Phase 1 — Stage 3 Normalization Constants

Responsibility
--------------
All tunable parameters and fixed labels for the normalization stage.
Single source of truth for filter.py and stage3.py.

Why these values are fixed
--------------------------
These are not user preferences — they are pyannote's requirements.
pyannote's segmentation and embedding models were trained at 16kHz mono.
Feeding any other configuration degrades all downstream output.
Do not make TARGET_SAMPLE_RATE or TARGET_CHANNELS configurable.

TARGET_LUFS is an EBU R128 broadcast standard value chosen to provide
consistent loudness across files while preserving dynamic range. It is
not a pyannote requirement but is forensic best practice.

Known biases
------------
KNOWN_BIAS_LABELS documents every transformation applied to the signal
during normalization. These are written verbatim to processing_notes
in the output JSON so that reviewers, researchers, and expert witnesses
can account for them.

The principle: document what was done to the signal, not just that
something was done. A future expert reading the output should be able
to reproduce the exact processing chain from these labels alone.

Contributors
------------
    Do not add logic here. Constants and labels only.
    If you add a new ffmpeg filter to filter.py, add its bias label here.
"""

# ---------------------------------------------------------------------------
# Normalization targets
# ---------------------------------------------------------------------------

#: pyannote requires exactly this sample rate. Non-negotiable.
TARGET_SAMPLE_RATE: int = 16_000

#: pyannote requires mono audio. Non-negotiable.
TARGET_CHANNELS: int = 1

#: EBU R128 integrated loudness target in LUFS.
#: -16.0 LUFS is standard for broadcast and forensic audio workflows.
TARGET_LUFS: float = -16.0

#: Output codec for the normalized WAV.
#: pcm_s16le is sufficient post-normalization — the working file used
#: pcm_f32le to preserve original precision for Stage 2 measurements.
#: After normalization the precision advantage of f32le is no longer
#: needed, and s16le halves the file size.
OUTPUT_CODEC: str = "pcm_s16le"

#: Output file suffix appended to the incident ID.
NORMALIZED_SUFFIX: str = "_normalized.wav"


# ---------------------------------------------------------------------------
# ffmpeg filter parameters
# ---------------------------------------------------------------------------

#: EBU R128 target integrated loudness for loudnorm filter.
LOUDNORM_TARGET_I: float = TARGET_LUFS

#: Maximum true peak level in dBTP. -1.5 leaves headroom below 0 dBFS.
LOUDNORM_TARGET_TP: float = -1.5

#: Loudness Range target in LU. 11.0 is the EBU R128 broadcast standard.
#: For BWC footage specifically, 11.0 LU is a deliberate choice: police
#: recordings often contain extreme dynamic range — a whispered command
#: followed immediately by a gunshot or shout. A target of 11.0 LU allows
#: the filter to compress enough to make quiet speech audible to pyannote
#: without completely flattening the dynamic character of loud events.
#: Tighter values (e.g. 7.0) would over-compress; looser values (e.g. 18.0)
#: would leave quiet speech too quiet for reliable diarization.
LOUDNORM_TARGET_LRA: float = 11.0

#: ffmpeg subprocess timeout in seconds.
#: Two-pass loudnorm is approximately real-time on modern CPU hardware,
#: meaning a 2-hour file takes roughly 2 hours to process in the worst
#: case. 3600 seconds (1 hour) covers typical BWC footage (15–90 min)
#: with headroom for slower hardware.
#:
#: For very long recordings (stakeout footage, 4+ hours), consider
#: computing a dynamic timeout in the orchestrator instead:
#:
#:   timeout = max(3600, int(props.duration_seconds * 2.5))
#:
#: The 2.5x multiplier accounts for two passes plus I/O overhead on
#: slow CPUs or files read from USB/external drives.
FFMPEG_TIMEOUT_SECONDS: int = 3600


# ---------------------------------------------------------------------------
# Known bias labels
# ---------------------------------------------------------------------------
#
# Each label is a short machine-readable key followed by a human-readable
# explanation. Written to processing_notes.stage_3.known_biases in output.
#
# Format: "KEY: explanation"

#: Downsampling from the original rate to 16kHz loses all frequency
#: content above 8kHz (Nyquist limit). Affected content: sibilant
#: consonants (s, sh, f, th) and some speaker-discriminating formants.
BIAS_HIGH_FREQ_LOSS: str = (
    "HIGH_FREQ_LOSS: Audio above 8kHz discarded during downsampling to "
    "16kHz (Nyquist limit). Sibilant consonants and high-frequency speaker "
    "features are not preserved. Speakers with similar fundamental "
    "frequencies but different high-frequency characteristics may be "
    "harder to distinguish in downstream diarization."
)

#: The anti-aliasing low-pass filter applied before downsampling slightly
#: attenuates frequencies in the 7–8kHz transition band.
BIAS_ANTIALIAS_ATTENUATION: str = (
    "ANTIALIAS_ATTENUATION: Low-pass anti-aliasing filter applied before "
    "downsampling attenuates the 7–8kHz frequency range. Effect is mild "
    "and most noticeable on high-pitched voices."
)

#: EBU R128 loudness normalization compresses the dynamic range of highly
#: variable audio to reach the -16 LUFS integrated target.
BIAS_LOUDNESS_NORMALIZATION: str = (
    "LOUDNESS_NORMALIZATION: EBU R128 two-pass loudnorm applied, targeting "
    f"{TARGET_LUFS} LUFS. Dynamic range may be compressed on recordings "
    "with highly variable loudness (e.g. quiet speech followed by shouting). "
    "RMS-per-speaker values from Stage 2 reflect pre-normalization energy "
    "and are not directly comparable to post-normalization values."
)

#: Stereo-to-mono downmix averages the two channels equally.
#: Only present when the source was stereo.
BIAS_MONO_DOWNMIX: str = (
    "MONO_DOWNMIX: Stereo channels averaged to mono "
    "(pan=mono|c0=0.5*c0+0.5*c1). Spatial audio information is lost. "
    "If one channel contained significantly different audio than the other "
    "(e.g. a dual-channel recording with officer and dispatch on separate "
    "channels), both are mixed equally into the mono output."
)