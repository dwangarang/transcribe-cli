#!/usr/bin/env python3
"""
transcribe.py — Transcribe any YouTube URL (video, playlist, or channel)

Strategy:
  1. Use youtube_transcript_api (instant, no download) — works if captions exist
  2. Fall back to yt-dlp audio download + faster-whisper (local, free)

Usage:
  python3 transcribe.py <youtube_url> [--workers N] [--output-dir DIR]
"""

import argparse
import os
import sys
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_video_ids(url: str) -> list[dict]:
    """Use yt-dlp to enumerate all videos from a URL (video, playlist, channel)."""
    print(f"Fetching video list from: {url}")
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching videos: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    videos = []
    for line in result.stdout.strip().splitlines():
        if "\t" in line:
            vid_id, title = line.split("\t", 1)
            videos.append({"id": vid_id, "title": title})
    return videos


def transcript_via_api(video_id: str) -> str | None:
    """Try to get transcript using youtube_transcript_api (no download needed)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return " ".join(snippet.text for snippet in transcript.snippets)
    except Exception:
        return None


def transcript_via_whisper(video_id: str, model_size: str = "base") -> str | None:
    """Download audio and transcribe with faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper not installed. Run: pip install faster-whisper", file=sys.stderr)
        return None

    audio_path = f"/tmp/{video_id}.mp3"
    try:
        # Download audio only
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "5",  # lower quality = faster download
            "-o", audio_path,
            "--no-playlist",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=1)
        return " ".join(seg.text for seg in segments)
    except Exception as e:
        print(f"  Whisper failed for {video_id}: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def process_video(video: dict, output_dir: Path, whisper_model: str) -> str:
    vid_id = video["id"]
    title = video["title"]
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
    out_file = output_dir / f"{safe_title}_{vid_id}.txt"

    if out_file.exists():
        return f"[SKIP] {title} (already exists)"

    # Try fast path first
    text = transcript_via_api(vid_id)
    method = "captions"

    if not text:
        text = transcript_via_whisper(vid_id, model_size=whisper_model)
        method = "whisper"

    if text:
        out_file.write_text(f"Title: {title}\nVideo ID: {vid_id}\n\n{text}\n", encoding="utf-8")
        return f"[OK:{method}] {title}"
    else:
        return f"[FAIL] {title}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos")
    parser.add_argument("url", help="YouTube video, playlist, or channel URL")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/Downloads"), help="Output directory (default: ~/Downloads)")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size for fallback (default: base)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    videos = get_video_ids(args.url)
    if not videos:
        print("No videos found.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s). Transcribing with {args.workers} workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_video, v, output_dir, args.whisper_model): v
            for v in videos
        }
        for future in as_completed(futures):
            print(future.result())

    print(f"\nDone. Transcripts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
