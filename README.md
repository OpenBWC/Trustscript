# TrustScript — Phase 1: Speaker Diarization Engine

> **Branch:** `phase-1/diarization`
> This branch implements the Speaker Diarization layer of the TrustScript pipeline.
> It will be merged into `main` once Phase 1 is complete and verified.

---

## What This Phase Covers

TrustScript is a forensic-grade transcription pipeline for police body-worn camera (BWC) footage. Rather than producing a transcript alone, it produces a transcript *plus* a structured record of confidence, uncertainty, and ambiguity at every level for interpretability.

**Phase 1 scope** (`src/diarization/`):

- Windowed diarization of long-form BWC audio (60+ min) without memory spikes
- Speaker embedding extraction via pyannote's ECAPA-TDNN/Community-1 model
- Speaker Anchor Vault — cross-chunk identity resolution to prevent permutation ambiguity
- Confidence scoring and ambiguity flagging on speaker assignments
- Output: a speaker timeline with IDs, embeddings, and confidence scores that feeds downstream phases

**Out of scope for this branch:**

| Module | Phase |
|---|---|
| `src/separation/` | Phase 3 — Overlapping speech separation (SepFormer/Conv-TasNet) |
| `src/asr/` | Phase 4 — Dual-engine ASR (Whisper + Kaldi) |
| `src/fusion/` | Phase 5 — Multi-source confidence fusion |
| `src/utils/schema.py` | Phase 6 — Final TrustScript JSON output formatting |

---

## System Requirements

| Requirement | Notes |
|---|---|
| Python 3.11 | 3.12+ not supported by pyannote.audio ecosystem |
| ffmpeg | System-level install required — not a pip package |
| RAM | 8GB minimum. Windowed processing keeps peak usage under 4GB |
| GPU | Not required. CPU-first by design; GPU support added in a later pass |

---

## System Dependencies

### ffmpeg

`pydub` (used for audio chunking) is a Python wrapper around ffmpeg. It must be installed at the OS level before any pip installs.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

---

## Installation

### Step 1 — Clone the repo and check out the branch

```bash
git clone https://github.com/OpenBWC/Trustscript.git
cd Trustscript
git checkout phase-1/diarization
```

### Step 2 — Create a Python 3.11 virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

Confirm the venv is using 3.11:
```bash
python --version
# Expected: Python 3.11.x
```

### Step 3 — Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4 — Install PyTorch (CPU build)

PyTorch must be installed separately before the rest of the requirements, pointing pip at the dedicated CPU wheel server. This guarantees the CPU-only build regardless of platform.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — HuggingFace authentication

pyannote.audio's model weights are hosted on HuggingFace behind a one-time license gate. You only need to do this once per machine.

**a) Accept the license agreements on HuggingFace** (requires a free account):
- pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0

**b) Authenticate your machine:**
```bash
huggingface-cli login
# Paste your HuggingFace token when prompted (read-only token is sufficient)
```

> **Note for offline/forensic deployments:** Once weights are downloaded they can be cloned
> locally via `git lfs` and loaded from a local path. pyannote does not phone home after
> the initial download. See the [Offline Deployment](#offline-deployment) section below.

---

## Verification

Run these checks after installation to confirm everything is wired correctly before writing any pipeline code.

### 1. Confirm CPU-only PyTorch build
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Expected:
```
2.11.0
False
```
`False` for `cuda.is_available()` is the critical confirmation. The version string may or may not include `+cpu` depending on how the wheel was resolved — the CUDA check is what matters.

### 2. Confirm pyannote and einops load cleanly
```bash
python -c "from pyannote.audio import Model; import einops; print('pyannote + einops OK')"
```
Expected:
```
pyannote + einops OK
```
> You may see a `UserWarning` about `torchcodec` — this is harmless. TrustScript loads
> audio via `librosa`/`pydub` and passes tensors directly to pyannote. torchcodec is
> never invoked in this pipeline. See the [Known Warnings](#known-warnings) section.

### 3. Confirm pydub can find ffmpeg
```bash
python -c "from pydub.utils import which; print('ffmpeg found at:', which('ffmpeg'))"
```
Expected:
```
ffmpeg found at: /opt/homebrew/bin/ffmpeg   # path will vary by OS
```
If this returns `ffmpeg found at: None`, ffmpeg is not on your PATH. Revisit the system dependency step.

### 4. Confirm full Phase 1 import chain
```bash
python -c "
import torch
import torchaudio
import pyannote.audio
import einops
import librosa
import soundfile
import pydub
import numpy
import scipy
import click
import tqdm
import rich
print('All Phase 1 dependencies OK')
"
```
Expected:
```
All Phase 1 dependencies OK
```

---

## Known Warnings

### torchcodec warning on pyannote import

When importing pyannote.audio, you may see a long `UserWarning` about `torchcodec` failing to load FFmpeg shared libraries. This is a known compatibility issue between torch 2.11.0 and Homebrew-installed FFmpeg on macOS ARM.

**This does not affect TrustScript.** The warning fires because torchcodec is pyannote's optional built-in audio decoder — but TrustScript never uses it. Audio is loaded externally via `librosa` and `pydub` in `src/utils/audio.py`, then passed to pyannote as in-memory tensors. torchcodec is never invoked.

This warning can be safely ignored for the duration of Phase 1.

---

## Offline Deployment

For forensic deployments where the processing machine must be air-gapped or cannot reach HuggingFace at runtime:

```bash
# After initial authentication, clone the model weights locally
git lfs install
git clone https://huggingface.co/pyannote/speaker-diarization-3.1 models/speaker-diarization-3.1
git clone https://huggingface.co/pyannote/segmentation-3.0 models/segmentation-3.0
```

Then load via local path in your pipeline code:
```python
pipeline = Pipeline.from_pretrained("models/speaker-diarization-3.1")
```

pyannote will not attempt any network calls when given a local path.

> **Privacy note for DOJ/law enforcement deployments:** Set the following environment
> variable before running any pipeline on real BWC footage to disable pyannote's
> usage telemetry:
> ```bash
> export PYANNOTE_METRICS_ENABLED=0
> ```
> Add this to your `.env` file or shell profile so it persists across sessions.

---

## Repository Structure (Phase 1 focus)

```
Trustscript/
├── src/
│   ├── diarization/           ← Phase 1 (this branch)
│   │   ├── engine.py          # Windowed diarization & pyannote logic
│   │   ├── vault.py           # Speaker Anchor Vault & centroid tracking
│   │   └── models.py          # Pydantic/dataclass models for diarization output
│   ├── utils/
│   │   └── audio.py           # Resampling, chunking, tensor prep
│   ├── separation/            # Phase 3 (stub — not implemented)
│   ├── asr/                   # Phase 4 (stub — not implemented)
│   ├── fusion/                # Phase 5 (stub — not implemented)
│   └── pipeline.py            # Orchestrator (stubbed for future phases)
├── models/                    # Downloaded .pth weights (git-ignored)
├── data/                      # Temporary chunk storage (git-ignored)
└── requirements.txt
```

---

## .gitignore

The following directories are git-ignored and should never be committed:

| Directory | Reason |
|---|---|
| `models/` | Downloaded model weights — large binary files, reproduced via `git lfs clone` |
| `data/` | Temporary audio chunks generated at runtime — may contain real BWC footage |

> **Important:** The `data/` directory may hold intermediate audio files from real
> police incidents during processing. Keeping it out of version control is both a
> storage concern and a chain-of-custody / privacy requirement.

---

## License

This project is part of [OpenBWC](https://github.com/OpenBWC), an open-source initiative
for transparent analysis of police body-worn camera footage.
See `LICENSE` for details.