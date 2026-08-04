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

It tries the cheap path first: if the video already has captions, `yt-dlp` fetches them instantly with no audio download. `youtube_transcript_api` is the second string, and only when both come up empty does it download the audio and run `faster-whisper` locally. Videos are processed in a thread pool (default 2 workers), and anything already transcribed is skipped, so re-running against a growing channel only does the new work.

Each transcript is written as `YYYY-MM-DD_Title_VIDEOID.txt`, upload date first, so an ordinary alphabetical file listing is also chronological. The same date is repeated in the file header.

```bash
yt <url> [--workers N] [--delay SECONDS] [--output-dir DIR]
         [--whisper-model tiny|base|small|medium]
         [--limit N] [--select newest|oldest|random] [--all] [--list]
```

**Scoping a playlist or channel.** A single video URL transcribes that one video and asks nothing. A playlist or channel URL reports how many videos it found and asks how much of it you want, answered as a count, a percentage, or `all`:

```
This page has 726 videos.
How many do you want to transcribe?
  a count (30), a percentage (25%), or 'all'. Enter defaults to all.
```

Pass the answer up front to skip the question, which is also what non-interactive runs need:

```bash
yt <page_url> --limit 30              # newest 30
yt <page_url> --limit 10%             # newest tenth
yt <page_url> --limit 10 --select oldest
yt <page_url> --list                  # count and list, transcribe nothing
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
- **Pull captions with `yt-dlp`, not the transcript API.** Fetching 73 videos through `youtube_transcript_api` got the IP blocked partway in: 23 succeeded, 50 returned `RequestBlocked`. The cause is the endpoint, not the volume. That library calls a legacy `timedtext` URL that YouTube throttles hard, while `yt-dlp` goes through the player API impersonating a real client. Same captions, far higher ceiling. Re-running the identical batch through `yt-dlp` finished 73 of 73 with no block. Parallelism and backoff were treated as the fix at first, and they were the wrong lever.
- **`json3` over `vtt` for captions.** Auto-generated `vtt` repeats each line in the following cue to produce the rolling-subtitle effect, so a naive parse duplicates most of the transcript. `json3` returns clean timestamped segments and needs no de-duplication.
- **`--sub-langs en,en-orig`, never `en.*`.** The wildcard also matches auto-translated tracks (`en-ar`, `en-es`, and so on). On a heavily translated video that turns one request into dozens, which trips the exact 429 the `yt-dlp` path exists to avoid. Worse, the resulting error aborted the call before the metadata file was written, so the upload date silently came back empty. A greedy pattern in one flag was undoing the fix in another.
- **Retry only what a retry can fix.** A rate-limit block and a video with no captions both surface as "no text". Retrying the first is correct and retrying the second wastes request budget, making the first more likely, so the error text is matched to tell them apart before any backoff.
- **Resume by video ID, not by filename.** Output files are named with the upload date, which isn't known until after the fetch. Matching an existing file on the ID instead keeps the skip-already-done behaviour working when a blocked batch is re-run.

## What you need before you start

Everything here is free. There is no paid service and no account with me.

| Requirement | Needed for | Notes |
|---|---|---|
| macOS on Apple Silicon | `transcribe` | The GPU path uses Metal. On an Intel Mac it falls back to CPU and runs roughly 5x slower. `yt` works anywhere Python does. |
| Python 3.9+ | both | Free |
| `ffmpeg`, `yt-dlp` | both | Free, via Homebrew |
| A [Hugging Face](https://huggingface.co) account | `transcribe` only | Free. You also have to accept the model terms, see step 3. |
| ~3 GB of disk | `transcribe` | Whisper and pyannote download on first run |

`yt` needs no account at all. If you only want YouTube transcripts, skip the Hugging Face steps entirely.

## Setup

**Requirements:** Python 3.9+, `ffmpeg`, `yt-dlp`.

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
