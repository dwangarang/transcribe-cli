"""Transcribe an audio file and label who is speaking.

Runs entirely on this Mac — audio is never uploaded anywhere. Two models do
the work: Whisper turns speech into text on the GPU, and pyannote works out
who spoke when, also on the GPU.

Called by the `transcribe` shell function in ~/.zshrc. Settings live in
~/.whisperx_config. The Hugging Face token is read from the login Keychain
by this script directly — never pass it as an argument, it would leak into
the log and into the process list.
"""

import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

MODEL_REPO = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
DIARIZATION_REPO = "pyannote/speaker-diarization-3.1"
SAMPLE_RATE = 16000


def get_hf_token():
    """Read the Hugging Face token from the environment or the login Keychain."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-a", os.environ.get("USER", ""),
             "-s", "huggingface_token", "-w"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return ""


def format_timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def load_pipeline(token):
    """Load the diarization pipeline, preferring the GPU.

    On the CPU this step runs about 5x slower than on the GPU and dominates
    the total runtime, so the GPU path matters a lot. Results are identical.
    """
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_REPO, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load {DIARIZATION_REPO}. The Hugging Face token may be "
            "invalid, or the model's terms may need accepting at "
            f"https://hf.co/{DIARIZATION_REPO}"
        )
    if torch.backends.mps.is_available():
        try:
            pipeline.to(torch.device("mps"))
            print("Diarization running on the GPU.")
            return pipeline
        except Exception as exc:
            print(f"GPU unavailable for diarization ({exc}); falling back to CPU.")
    else:
        print("GPU unavailable for diarization; using CPU (slower).")
    return pipeline


def assign_speakers(segments, turns):
    """Attach a speaker to each transcript segment, then merge runs."""
    labeled = []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        text = segment["text"].strip()
        if not text:
            continue

        # Total overlap per speaker, so someone split across several short
        # turns still beats one long turn that barely touches the segment.
        overlap_by_speaker = defaultdict(float)
        for turn_start, turn_end, speaker in turns:
            overlap = min(turn_end, end) - max(turn_start, start)
            if overlap > 0:
                overlap_by_speaker[speaker] += overlap

        if overlap_by_speaker:
            speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
        elif labeled:
            speaker = labeled[-1][0]
        else:
            speaker = "SPEAKER_UNKNOWN"

        labeled.append((speaker, text, start))

    blocks = []
    for speaker, text, start in labeled:
        if blocks and blocks[-1][0] == speaker:
            blocks[-1][1].append(text)
        else:
            blocks.append((speaker, [text], start))
    return blocks


def transcribe(audio_file, output_dir, min_speakers, max_speakers, language):
    import numpy as np
    import torch
    import mlx_whisper
    from mlx_whisper.audio import load_audio

    token = get_hf_token()
    if not token:
        print("ERROR: No Hugging Face token found.")
        print("Store one with:")
        print('  security add-generic-password -a "$USER" -s huggingface_token -w')
        return 1

    # Decode the audio once and reuse it for both models, rather than having
    # each of them shell out to ffmpeg and re-read the file.
    print("Loading audio...")
    audio = np.array(load_audio(audio_file, sr=SAMPLE_RATE), copy=True)
    duration = len(audio) / SAMPLE_RATE
    print(f"Length: {format_timestamp(duration)}")

    print(f"Step 1/3: Transcribing ({language or 'auto-detect'}) with {MODEL_REPO.split('/')[-1]}...")
    started = time.time()
    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL_REPO, language=language, verbose=False
    )
    print(f"  done in {time.time() - started:.0f}s — language used: {result.get('language')}")

    print("Step 2/3: Identifying speakers...")
    started = time.time()
    pipeline = load_pipeline(token)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    diarization = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        min_speakers=int(min_speakers),
        max_speakers=int(max_speakers),
    )
    turns = [(t.start, t.end, spk)
             for t, _, spk in diarization.itertracks(yield_label=True)]
    speakers = sorted({spk for _, _, spk in turns})
    print(f"  done in {time.time() - started:.0f}s — found {len(speakers)} speaker(s)")

    print("Step 3/3: Combining transcript and speakers...")
    blocks = assign_speakers(result.get("segments", []), turns)

    lines = [f"[{format_timestamp(start)}] {speaker}: {' '.join(parts)}"
             for speaker, parts, start in blocks]

    basename = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_dir, f"{basename}.txt")
    with open(output_file, "w") as f:
        f.write("\n\n".join(lines) + "\n")

    print(f"Saved: {output_file}")
    return 0


if __name__ == "__main__":
    audio_file = sys.argv[1]
    output_dir = sys.argv[2]
    min_speakers = sys.argv[3]
    max_speakers = sys.argv[4]
    language = sys.argv[5] if len(sys.argv) > 5 else "auto"
    if language.lower() in ("auto", "none", ""):
        language = None
    sys.exit(transcribe(audio_file, output_dir, min_speakers, max_speakers, language))
