"""
src/cli/probe.py
================
TrustScript CLI — Sandbox Tool for Raw Model Inspection

Responsibility
--------------
Executes TrustScript preprocessing (Stages 1-4) to safely extract and 
normalize audio (supporting .mp4, etc.), then bypasses the Phase 1 
Engine entirely. 

Runs the raw pyannote community-1 model against the normalized audio 
to output the unwrapped Annotation and raw segmentation probabilities.
Used purely for developer debugging to understand base model behavior.
"""

import logging
from pathlib import Path
import sys

import click
from dotenv import load_dotenv

from .main import configure_logging, console


@click.command("probe")
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a single BWC video/audio file.",
)
@click.option(
    "--token",
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace access token.",
)
@click.option(
    "--models-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Path to local pyannote weights.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def probe(input_path: str, token: str | None, models_dir: str | None, verbose: bool) -> None:
    """
    Developer Sandbox: Run raw pyannote inference on an audio/video file.

    Executes TrustScript Stages 1-4 to normalize the input, then runs 
    a single full-file inference pass to print the raw Pyannote V4 
    objects and segmentation probabilities.
    """
    load_dotenv()
    configure_logging(verbose, None)
    log = logging.getLogger(__name__)

    file = Path(input_path)
    incident_id = file.stem
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]PROBE SANDBOX: {file.name}[/bold]")

    try:
        # --- STAGE 1: Format Detection ---
        console.print("[dim]Running Stage 1 (Input Handling)...[/dim]")
        from src.utils.audio import run_stage1
        stage1_result = run_stage1(file)

        # --- AUDIO EXTRACTION ---
        console.print("[dim]Extracting audio...[/dim]")
        from src.utils.audio.extraction import extract_working_audio
        working_audio_path, _, _ = extract_working_audio(
            stage1_result.original_path, output_dir, incident_id
        )

        # --- STAGE 2: Profiling ---
        console.print("[dim]Running Stage 2 (Quality Profiling)...[/dim]")
        from src.utils.audio import run_stage2
        stage2_result = run_stage2(stage1_result, working_audio_path)

        # --- STAGE 3: Normalization ---
        console.print("[dim]Running Stage 3 (16kHz Mono Normalization)...[/dim]")
        from src.utils.normalization import run_stage3
        stage3_result = run_stage3(
            stage1_result, working_audio_path, output_dir, incident_id
        )
        normalized_wav = stage3_result.normalized_path

        # --- STAGE 4: Model Loading ---
        console.print("[dim]Running Stage 4 (Loading pyannote)...[/dim]")
        from src.diarization.models import run_stage4
        stage4_result = run_stage4(token=token, models_dir=Path(models_dir) if models_dir else None)
        pipeline = stage4_result.pipeline

    except Exception as e:
        console.print(f"[red]Preprocessing failed: {e}[/red]")
        sys.exit(1)

   # ------------------------------------------------------------------
    # SANDBOX INFERENCE
    # ------------------------------------------------------------------
    console.print("\n[bold yellow]Preprocessing complete. Starting full-file inference...[/bold yellow]")
    console.print("[dim](This may take several minutes depending on file length and hardware)[/dim]")

    try:
        import soundfile as sf
        import torch
        
        # 1. Load the entire file using soundfile to bypass torchaudio/torchcodec bugs
        data, sample_rate = sf.read(str(normalized_wav), dtype="float32", always_2d=True)
        
        # 2. Convert to PyTorch Tensor shape (channels, samples)
        waveform = torch.from_numpy(data.T)
        
        # 3. Create the dictionary input Pyannote expects
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        # 4. Run standard diarization
        raw_output = pipeline(audio_input)
    except Exception as e:
        console.print(f"[red]Inference failed: {e}[/red]")
        sys.exit(1)

    # Unwrap DiarizeOutput
    console.print("\n[bold cyan]=== BASE DIARIZATION OUTPUT (Unwrapped Annotation) ===[/bold cyan]")
    diarization = getattr(raw_output, "speaker_diarization", raw_output)
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        console.print(f"  [{turn.start:06.2f}s -> {turn.end:06.2f}s] {speaker}")

   # Extract Raw Probabilities
    console.print("\n[bold cyan]=== RAW SEGMENTATION PROBABILITIES ===[/bold cyan]")
    try:
        # 1. Get the raw scores from the internal model
        posteriors = pipeline._segmentation(audio_input)
        
        # 2. Handle both SlidingWindowFeature and raw Tensors
        data = getattr(posteriors, "data", posteriors)
        
        # 3. Robust shape unpacking (Taking only the first two dimensions)
        # This prevents the "too many values to unpack" error if data is 3D
        shape = data.shape
        frames = shape[0]
        speakers = shape[1]
        
        console.print(f"  [green]Success:[/green] Matrix shape: {shape}")
        console.print("  [dim]Showing first 25 frames where max probability > 0.1:[/dim]\n")
        
        # If it's a torch tensor, move to CPU and convert to numpy for printing
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()

        printed = 0
        for i in range(frames):
            # If 3D (frames, speakers, 1), we flatten to 2D for the logic below
            frame = data[i].flatten() 
            
            if frame.max() > 0.1:
                probs = ", ".join([f"Spk{j}: {p:.3f}" for j, p in enumerate(frame)])
                
                # Check for overlap (Sum of top 2 probabilities > 1.0)
                # This is the "Powerset" logic in v4
                sorted_probs = sorted(frame, reverse=True)
                is_overlap = len(sorted_probs) > 1 and (sorted_probs[0] + sorted_probs[1] > 1.0)
                overlap_flag = " [red]<-- OVERLAP[/red]" if is_overlap else ""
                
                console.print(f"  Frame {i:05d}: {probs}{overlap_flag}")
                printed += 1
            if printed >= 25:
                break

    except Exception as e:
        console.print(f"[red]Failed to extract probabilities: {e}[/red]")
        # Log the actual shape to help you debug exactly what v4 returned
        if 'data' in locals():
            console.print(f"[dim]Attempted to unpack shape: {data.shape}[/dim]")