# transcribe-cli

Two command-line tools for turning speech into searchable text on a Mac, plus the shell glue that makes them one-word commands.

```bash
yt https://youtube.com/watch?v=...      # YouTube video, playlist, or channel → transcripts
transcribe ~/Downloads/meeting.m4a      # local audio → transcript with speaker labels
```

Both run locally. The audio-file path never uploads anything — Whisper and the speaker-diarization model both run on the Mac's GPU.

## The two tools

### `yt` — YouTube transcripts

`transcribe_youtube.py` takes any YouTube video, playlist, or channel URL and writes one text file per video.

It tries the cheap path first: if the video already has captions, `youtube_transcript_api` fetches them instantly with no download. Only when that fails does it fall back to downloading the audio with `yt-dlp` and running `faster-whisper` locally. Videos are processed in a thread pool (default 4 workers), and anything already transcribed is skipped, so re-running against a growing channel only does the new work.

```bash
yt <url> [--workers N] [--output-dir DIR] [--whisper-model tiny|base|small|medium]
```

### `transcribe` — local audio with speaker labels

`transcribe_audio.py` handles recordings where *who said it* matters — interviews, meetings, calls. Output looks like:

```
[00:00:12] SPEAKER_01: So the question is whether we ship this quarter.

[00:00:19] SPEAKER_00: I don't think the numbers support that yet.
```

It runs two models: `mlx-whisper` for speech-to-text and `pyannote` for diarization, both on the GPU via Metal, with an automatic CPU fallback.

```bash
transcribe /path/to/audio [language]   # language: en, zh, auto
logwatch                               # follow progress of a running job
```

Long files run in the background under `nohup`, hold `caffeinate` so the Mac won't sleep mid-job, and fire a native notification when they finish. You can close the terminal.

## Implementation notes

The parts that took the most thought:

- **The token never touches the shell.** The Hugging Face token is read directly from the macOS Keychain inside the Python process, not exported as an env var or passed as an argument. Arguments show up in `ps` output and get written into the log in plain text; this avoids both.
- **Decode the audio once.** Whisper and pyannote each want the same waveform, and each would happily shell out to `ffmpeg` to get it. Loading it once into a numpy array and handing it to both roughly halves the I/O on long files.
- **Overlap-weighted speaker assignment.** Naively tagging a transcript segment with whichever diarization turn it starts inside gets crosstalk wrong. `assign_speakers` instead totals overlap duration per speaker across all turns touching the segment, so someone split across several short turns still wins over one long turn that barely grazes it.
- **Don't leave `auto` on.** Whisper guesses language from the first 30 seconds and has confidently misread accented English as Nynorsk. The config defaults to an explicit language with a per-run override.
- **`noglob` on the `yt` alias.** YouTube URLs contain `?` and `&`, which zsh will otherwise try to expand into filenames.

## Setup

**Requirements:** macOS on Apple Silicon (for the GPU path), Python 3.9+, `ffmpeg`, `yt-dlp`.

```bash
brew install ffmpeg yt-dlp
pip install mlx-whisper pyannote.audio torch numpy      # for `transcribe`
pip install youtube-transcript-api faster-whisper       # for `yt`
```

**1. Clone and wire up the shell:**

```bash
git clone https://github.com/dwangarang/transcribe-cli.git ~/transcribe-cli
```

Add to `~/.zshrc`:

```bash
export TRANSCRIBE_CLI_HOME="$HOME/transcribe-cli"
source "$TRANSCRIBE_CLI_HOME/shell/functions.zsh"
```

**2. Create the config:**

```bash
cp ~/transcribe-cli/whisperx_config.example ~/.whisperx_config
```

Edit it to set your language, expected speaker count, and output directory.

**3. Store a Hugging Face token in the Keychain.** The diarization model is gated — create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the terms at [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1), then:

```bash
security add-generic-password -a "$USER" -s huggingface_token -w
```

Restart your shell and both commands are live.

## License

MIT
