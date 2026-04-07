"""
src/diarization/models.py
==========================
TrustScript Phase 1 — Stage 4: Model Loading

Responsibility
--------------
Load pyannote's two required models onto CPU and return them as a
typed result. Models are loaded once per pipeline run — not once per
chunk. Passing the loaded models forward avoids the 5–15 second
initialisation cost on every diarization window.

Models loaded
-------------
    pyannote/speaker-diarization-community-1
        The full diarization pipeline. Uses the Community-1 weights
        which include the powerset segmentation architecture. This is
        what detects "who spoke when" and identifies overlapping speech
        as a distinct class rather than misassigning it.

    pyannote/embedding
        ECAPA-TDNN speaker embedding model. Extracts 512-dimensional
        voice fingerprints from audio segments. Used by the Speaker
        Anchor Vault in Stage 5 for cosine similarity matching across
        chunks.

CPU configuration
-----------------
Both models are explicitly moved to CPU. TrustScript is CPU-first by
design — GPU support is added in a later pass. This ensures the
pipeline runs on any machine without CUDA dependencies.

    embedding_batch_size=1: limits RAM usage to ~2–3GB per chunk,
    keeping the pipeline viable on 8GB machines.

Telemetry
---------
pyannote.audio v4 includes optional usage telemetry. For a pipeline
processing real police footage under a DOJ partnership, telemetry must
be disabled. PYANNOTE_METRICS_ENABLED=0 is set in the environment
before any pyannote import resolves.

Token resolution
----------------
HuggingFace authentication is required to download gated model weights
on first use. After first download, weights are cached locally and no
token is needed for subsequent runs.

Resolution order:
    1. token argument passed directly to this module
    2. HF_TOKEN environment variable (set via .env or shell)
    3. Hard error with actionable setup instructions

Offline / air-gapped deployment
---------------------------------
Pass a local directory path via models_dir to load weights from disk
without any HuggingFace network calls. See README.md — Offline
Deployment for setup instructions.

Contributors
------------
    Do not add diarization logic here. This module only loads models.
    Engine logic (chunking, vault matching) belongs in engine.py.
    Vault logic belongs in vault.py.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Disable pyannote telemetry before any pyannote import.
# Must be set before the first import of any pyannote module.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HuggingFace model identifier for the diarization pipeline.
DIARIZATION_MODEL_ID: str = "pyannote/speaker-diarization-community-1"

#: HuggingFace model identifier for the speaker embedding model.
EMBEDDING_MODEL_ID: str = "pyannote/embedding"

#: Embedding batch size. Set to 1 to keep RAM under 4GB on 8GB machines.
#: Increasing this speeds up embedding extraction at the cost of higher
#: peak RAM. Do not increase above 4 without profiling memory usage.
EMBEDDING_BATCH_SIZE: int = 1

#: Inference window for the embedding model.
#: "whole" extracts one embedding per segment (correct for vault use).
#: "sliding" would extract multiple embeddings per segment.
EMBEDDING_WINDOW: str = "whole"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Stage4Result:
    """
    Complete output of Stage 4: Model Loading.

    Holds the loaded pyannote models for use throughout Stage 5.
    Both models are on CPU and ready for inference.

    Attributes
    ----------
    pipeline : object
        Loaded pyannote Pipeline instance
        (pyannote.audio.Pipeline). Used for windowed diarization in
        Stage 5. Type is object to avoid importing pyannote at the
        module level in files that only hold this result.
    embed_inference : object
        Loaded pyannote Inference instance
        (pyannote.audio.Inference). Used for speaker embedding
        extraction in Stage 5's vault matching loop.
    diarization_model_id : str
        HuggingFace model ID or local path used for the pipeline.
        Written to output metadata for reproducibility.
    embedding_model_id : str
        HuggingFace model ID or local path used for embeddings.
        Written to output metadata for reproducibility.
    loaded_from_local : bool
        True when models were loaded from a local directory (offline
        deployment). False when loaded from HuggingFace cache.
    """
    pipeline: object
    embed_inference: object
    diarization_model_id: str
    embedding_model_id: str
    loaded_from_local: bool

    def to_processing_notes(self) -> dict:
        """
        Serialize Stage 4 metadata for the processing_notes block.
        Called by Stage 7 (Output Assembly).
        """
        return {
            "stage_4": {
                "diarization_model": self.diarization_model_id,
                "embedding_model": self.embedding_model_id,
                "loaded_from_local": self.loaded_from_local,
                "device": "cpu",
                "embedding_batch_size": EMBEDDING_BATCH_SIZE,
                "telemetry_disabled": True,
            }
        }


# ---------------------------------------------------------------------------
# Stage 4 entry point
# ---------------------------------------------------------------------------


def run_stage4(
    token: str | None = None,
    models_dir: Path | None = None,
) -> Stage4Result:
    """
    Execute Stage 4: Model Loading.

    Loads the pyannote diarization pipeline and embedding inference
    model onto CPU. Returns a Stage4Result holding both for use in
    Stage 5.

    Models are loaded once per run. Pass the returned Stage4Result
    to run_stage5() — do not call run_stage4() inside the chunking loop.

    Parameters
    ----------
    token : str | None
        HuggingFace access token. If None, falls back to the HF_TOKEN
        environment variable. If neither is available and models are not
        locally cached, raises RuntimeError with setup instructions.
    models_dir : Path | None
        Path to a local directory containing pre-downloaded model weights.
        Use for offline or air-gapped deployments. When provided, token
        is not required. See README.md — Offline Deployment.

    Returns
    -------
    Stage4Result
        Loaded pipeline and embedding inference, ready for Stage 5.

    Raises
    ------
    RuntimeError
        If models cannot be loaded — HuggingFace auth failed, models
        not found locally, or pyannote import errors.
    """
    # Lazy import — pyannote is only imported when this function runs.
    # This keeps startup time fast for --help and other CLI commands
    # that don't need the models.
    try:
        from pyannote.audio import Inference, Model, Pipeline
    except ImportError as e:
        raise RuntimeError(
            "pyannote.audio is not installed. "
            "Run: pip install pyannote.audio\n"
            f"Original error: {e}"
        ) from e

    logger.info("Stage 4 — Model Loading")

    # Resolve the token from argument or environment.
    resolved_token = token or os.environ.get("HF_TOKEN")

    # ------------------------------------------------------------------
    # Determine model source: local directory or HuggingFace.
    # ------------------------------------------------------------------
    if models_dir is not None:
        diarization_source = str(
            models_dir / "pyannote-speaker-diarization-community-1"
        )
        embedding_source = str(models_dir / "pyannote-embedding")
        loaded_from_local = True
        logger.info(
            "Loading models from local directory: %s", models_dir
        )
    else:
        diarization_source = DIARIZATION_MODEL_ID
        embedding_source = EMBEDDING_MODEL_ID
        loaded_from_local = False

        if not resolved_token:
            raise RuntimeError(
                "No HuggingFace token found.\n"
                "TrustScript requires a token to download pyannote models.\n\n"
                "Resolution options:\n"
                "  1. Add HF_TOKEN=your_token to your .env file\n"
                "  2. Pass --token hf_xxx to the CLI\n"
                "  3. Run: huggingface-cli login\n\n"
                "To accept model license agreements (required once):\n"
                f"  https://huggingface.co/{DIARIZATION_MODEL_ID}\n"
                f"  https://huggingface.co/pyannote/segmentation-3.0\n"
                f"  https://huggingface.co/{EMBEDDING_MODEL_ID}\n\n"
                "For offline/air-gapped deployments, see README.md — "
                "Offline Deployment, then pass --models-dir to the CLI."
            )
        logger.info("Loading models from HuggingFace (token present).")

    # ------------------------------------------------------------------
    # Load diarization pipeline.
    # ------------------------------------------------------------------
    logger.info("Loading diarization pipeline: %s", diarization_source)
    try:
        pipeline = Pipeline.from_pretrained(
            diarization_source,
            token=resolved_token,
        )
        pipeline.to(torch.device("cpu"))

        # Limit embedding batch size for RAM safety on 8GB machines.
        pipeline.embedding_batch_size = EMBEDDING_BATCH_SIZE

        logger.debug(
            "Diarization pipeline loaded. "
            "embedding_batch_size=%d, device=cpu",
            EMBEDDING_BATCH_SIZE,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load diarization pipeline '{diarization_source}'.\n"
            f"Error: {e}\n\n"
            "If this is an authentication error, ensure you have:\n"
            "  1. A valid HuggingFace token\n"
            "  2. Accepted the model license at "
            f"https://huggingface.co/{DIARIZATION_MODEL_ID}"
        ) from e

    # ------------------------------------------------------------------
    # Load embedding inference model.
    # ------------------------------------------------------------------
    logger.info("Loading embedding model: %s", embedding_source)
    try:
        embedding_model = Model.from_pretrained(
            embedding_source,
            token=resolved_token,
        )
        embed_inference = Inference(
            embedding_model,
            window=EMBEDDING_WINDOW,
            device=torch.device("cpu"),
        )
        logger.debug("Embedding model loaded. window=%s", EMBEDDING_WINDOW)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{embedding_source}'.\n"
            f"Error: {e}\n\n"
            "If this is an authentication error, ensure you have:\n"
            "  1. A valid HuggingFace token\n"
            "  2. Accepted the model license at "
            f"https://huggingface.co/{EMBEDDING_MODEL_ID}"
        ) from e

    logger.info(
        "Stage 4 complete — both models loaded on CPU. "
        "Telemetry disabled (PYANNOTE_METRICS_ENABLED=0)."
    )

    return Stage4Result(
        pipeline=pipeline,
        embed_inference=embed_inference,
        diarization_model_id=diarization_source,
        embedding_model_id=embedding_source,
        loaded_from_local=loaded_from_local,
    )