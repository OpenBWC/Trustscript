"""
src/cli/probe.py
================
TrustScript CLI — Sandbox Tool for Raw Model Inspection

Owns:
    - The `trustscript probe` Click command.

Responsibility
--------------
Bypass the TrustScript Engine, Vault, and Gate system entirely. 
Loads the raw pyannote community-1 model and runs it against a target
audio file. Used purely for developer debugging to understand base
model behavior, API changes, and raw probability outputs.
"""

import logging
from pathlib import Path

import click
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

from .main import console


@click.command("probe")
@click.argument(
    "audio_path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--token",
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace access token.",
)
def probe_model(audio_path: str, token: str | None) -> None:
    """
    Developer Sandbox: Run raw pyannote inference on an audio file.

    Bypasses the TrustScript Phase 1 pipeline entirely to output the
    raw underlying Pyannote objects, including the unwrapped Annotation
    and the raw segmentation probabilities.
    """
    load_dotenv()
    
    # 1. Hardware Detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print("[green]Hardware acceleration (Apple MPS) active.[/green]")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        console.print("[green]Hardware acceleration (CUDA) active.[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]No hardware acceleration. Using CPU.[/yellow]")

    # 2. Load Model
    console.print(f"Loading pyannote/speaker-diarization-community-1...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=token,
        )
        pipeline.to(device)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return

    # 3. Inference
    console.print(f"Running inference on {Path(audio_path).name}...")
    try:
        # Note: For this raw test, we can pass the filepath directly to pyannote
        # rather than building the waveform dictionary we need for the main engine.
        raw_output = pipeline(audio_path)
    except Exception as e:
        console.print(f"[red]Inference failed: {e}[/red]")
        return

    # 4. Unwrap the DiarizeOutput (The v4 API shift)
    console.print("\n[bold cyan]=== BASE DIARIZATION OUTPUT ===[/bold cyan]")
    diarization = getattr(raw_output, "speaker_diarization", raw_output)
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        console.print(f"  [{turn.start:05.2f}s -> {turn.end:05.2f}s] {speaker}")

    # 5. Extract Raw Probabilities (The Posteriors API shift)
    console.print("\n[bold cyan]=== RAW SEGMENTATION PROBABILITIES ===[/bold cyan]")
    try:
        # This calls the internal segmentation model directly to get the raw float math
        posteriors = pipeline._segmentation(audio_path)
        
        # posteriors.data is a numpy array of shape (num_frames, num_speakers)
        data = posteriors.data
        frames, speakers = data.shape
        
        console.print(f"  Shape: {frames} frames x {speakers} speakers detected.")
        console.print("  [dim](Each frame represents ~16ms of audio. Values are probabilities 0.0 to 1.0)[/dim]\n")
        
        # Print a sample of the first 10 frames that contain actual speech
        printed = 0
        for i, frame in enumerate(data):
            # Only print frames where at least one speaker has > 10% probability
            if frame.max() > 0.1:
                probs = ", ".join([f"Spk{j}: {p:.2f}" for j, p in enumerate(frame)])
                console.print(f"  Frame {i:04d}: {probs}")
                printed += 1
            if printed >= 10:
                break
                
    except Exception as e:
        console.print(f"[red]Failed to extract probabilities: {e}[/red]")