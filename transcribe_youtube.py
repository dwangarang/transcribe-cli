#!/usr/bin/env python3
"""
transcribe.py — Transcribe any YouTube URL (video, playlist, or channel)

Strategy:
  1. Pull captions with yt-dlp (instant, no audio download)
  2. Fall back to youtube_transcript_api if yt-dlp comes up empty
  3. Fall back to yt-dlp audio download + faster-whisper (local, free)

On rate limiting:
  YouTube throttles the legacy timedtext endpoint that youtube_transcript_api
  uses, which is why yt-dlp is tried first. On top of that the script pauses a
  jittered --delay before each request, backs off exponentially when it does get
  blocked, and skips videos whose output file already exists — so re-running the
  same command after a blocked batch retries only what failed.

Usage:
  python3 transcribe.py <youtube_url> [--workers N] [--output-dir DIR]

Scoping a playlist or channel:
  A single video URL always transcribes that one video, no questions asked.
  A playlist or channel URL reports how many videos it found and asks how many
  of them you want. Answer with a count (30), a percentage (25%), or "all".
  To skip the question, pass the answer up front:

  python3 transcribe.py <page_url> --limit 30       # first 30 videos
  python3 transcribe.py <page_url> --limit 25%      # first quarter of them
  python3 transcribe.py <page_url> --limit 10 --select oldest
  python3 transcribe.py <page_url> --all            # every video
  python3 transcribe.py <page_url> --list           # just count and list, no transcribing
"""

import argparse
import math
import os
import random
import re
import sys
import subprocess
import tempfile
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_warned_no_whisper = False


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


def parse_limit(spec: str, total: int) -> int:
    """Turn an answer like '30', '25%', or 'all' into a concrete video count.

    Percentages round up, so '1%' of 20 videos still gives you 1 rather than 0.
    The result is clamped to the list you actually have: asking for 500 videos
    from a 40-video channel gives you 40, not an error.
    Raises ValueError on anything that isn't a usable amount.
    """
    spec = spec.strip().lower()
    if spec in ("", "all", "a"):
        return total

    if spec.endswith("%"):
        try:
            pct = float(spec[:-1].strip())
        except ValueError:
            raise ValueError(f"'{spec}' is not a percentage")
        if pct <= 0:
            raise ValueError("percentage must be greater than 0")
        return max(1, min(total, math.ceil(total * pct / 100)))

    try:
        count = int(spec)
    except ValueError:
        raise ValueError(f"'{spec}' is not a number or a percentage")
    if count <= 0:
        raise ValueError("count must be greater than 0")
    return min(total, count)


def order_videos(videos: list[dict], select: str) -> list[dict]:
    """Decide which end of the list survives the cut.

    yt-dlp lists channels and playlists newest-first, so 'newest' is just the
    order it handed us and costs nothing.
    """
    if select == "oldest":
        return list(reversed(videos))
    if select == "random":
        shuffled = list(videos)
        random.shuffle(shuffled)
        return shuffled
    return list(videos)


def prompt_for_limit(total: int) -> str:
    """Ask how much of the page to transcribe. Re-asks until the answer parses."""
    print(f"\nThis page has {total} videos.")
    print("How many do you want to transcribe?")
    print("  a count (30), a percentage (25%), or 'all'. Enter defaults to all.")
    while True:
        try:
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        try:
            parse_limit(answer, total)
            return answer
        except ValueError as e:
            print(f"  {e}. Try again.")


def scope_videos(videos: list[dict], limit: str | None, select: str,
                 take_all: bool) -> list[dict]:
    """Cut a page's video list down to the requested amount.

    A single video is returned untouched and is never prompted about, so
    `yt <video_url>` behaves exactly as it always has.
    """
    total = len(videos)
    if total <= 1 or take_all:
        return videos

    if limit is None:
        if not sys.stdin.isatty():
            # Piped or scripted run with no --limit: don't hang on input().
            print(f"Found {total} videos and no --limit given — taking all of them.")
            return videos
        limit = prompt_for_limit(total)

    count = parse_limit(limit, total)
    selected = order_videos(videos, select)[:count]

    if count < total:
        print(f"\nScoped to {count} of {total} videos ({select} first):")
        for video in selected[:5]:
            print(f"  - {video['title']}")
        if count > 5:
            print(f"  ... and {count - 5} more")
    return selected


BLOCK_SIGNS = (
    "429", "too many requests", "sign in to confirm", "not a bot",
    "blocked", "http error 403", "requestblocked", "rate limit",
)


def looks_like_block(text: str) -> bool:
    """Distinguish 'YouTube is throttling us' from 'this video has no captions'.

    Worth the effort: a block should be retried after a pause, while a video
    with no captions should be given up on immediately. Retrying the second
    case just burns request budget and makes the first case more likely.
    """
    low = text.lower()
    return any(sign in low for sign in BLOCK_SIGNS)


def parse_json3(path: Path) -> str:
    """Flatten YouTube's json3 caption format into one block of text."""
    data = json.loads(path.read_text(encoding="utf-8"))
    words = []
    for event in data.get("events", []):
        for seg in event.get("segs") or []:
            words.append(seg.get("utf8", ""))
    return " ".join("".join(words).split())


def format_upload_date(raw: str | None) -> str:
    """Turn yt-dlp's '20260804' into '2026-08-04'.

    ISO order is what makes the filename sort chronologically, since a plain
    alphabetical sort of YYYY-MM-DD is also a date sort. 'undated' sorts after
    any digit, so unknowns collect at the end of the folder rather than the top.
    """
    if raw and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return "undated"


def transcript_via_ytdlp(video_id: str, delay: float,
                         retries: int = 3) -> tuple[str | None, str | None]:
    """Fetch captions AND upload date through yt-dlp rather than the timedtext API.

    This is the whole rate-limit fix. youtube_transcript_api calls a legacy
    endpoint that YouTube throttles hard; yt-dlp goes through the player API
    impersonating a real client, which has a far higher ceiling. Verified: the
    API returned RequestBlocked for a video that yt-dlp fetched without issue.

    json3 rather than vtt because auto-generated vtt repeats each line in the
    following cue, which would need de-duplicating; json3 does not.

    --write-info-json rides along on the same request, so the upload date costs
    no extra network round trip. Returns (text, raw_date); either may be None,
    and the date can survive even when there are no captions.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(retries):
        # Jittered pause before every request. Jitter matters because the
        # worker threads would otherwise re-sync into bursts after each wait.
        time.sleep(random.uniform(delay, delay * 2))

        blocked = False
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "yt-dlp",
                "--write-subs", "--write-auto-subs",
                # "en,en-orig" and NOT "en.*": the wildcard also matches the
                # auto-translated tracks (en-ar, en-es, ...), which on a
                # heavily translated video means dozens of extra requests per
                # video. That both wastes request budget and trips the 429 this
                # function exists to avoid.
                "--sub-langs", "en,en-orig",
                "--sub-format", "json3",
                "--write-info-json",
                # Without this a failed subtitle track aborts the whole call
                # before the info json is written, losing the upload date.
                "--ignore-errors",
                "--skip-download", "--no-warnings", "--no-playlist",
                "-o", f"{tmpdir}/%(id)s.%(ext)s",
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            raw_date = None
            info_files = list(Path(tmpdir).glob("*.info.json"))
            if info_files:
                try:
                    info = json.loads(info_files[0].read_text(encoding="utf-8"))
                    raw_date = info.get("upload_date")
                except (json.JSONDecodeError, OSError):
                    pass

            files = list(Path(tmpdir).glob("*.json3"))
            if files:
                # Prefer the manual/primary track over the "-orig" auto track.
                files.sort(key=lambda p: ".en-orig." in p.name)
                text = parse_json3(files[0])
                if text:
                    return text, raw_date

            blocked = looks_like_block(result.stderr + result.stdout)

        if not blocked:
            # Genuinely no captions, so retrying would be pointless. Still hand
            # back the date so the fallback paths can name the file properly.
            return None, raw_date

        backoff = 5 * (2 ** attempt) + random.uniform(0, 3)
        print(f"  rate-limited on {video_id}, waiting {backoff:.0f}s before retry "
              f"({attempt + 1}/{retries})", file=sys.stderr)
        time.sleep(backoff)

    return None, None


def transcript_via_api(video_id: str) -> str | None:
    """Secondary caption path via youtube_transcript_api (no download needed).

    Kept as a backstop for the rare video yt-dlp cannot pull, but it is no
    longer tried first: it is the endpoint that gets blocked.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return " ".join(snippet.text for snippet in transcript.snippets)
    except Exception:
        return None


def transcript_via_whisper(video_id: str, model_size: str = "base") -> str | None:
    """Download audio and transcribe with faster-whisper."""
    global _warned_no_whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        if not _warned_no_whisper:  # once per run, not once per video
            print("faster-whisper not installed. Run: pip install faster-whisper", file=sys.stderr)
            _warned_no_whisper = True
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


def process_video(video: dict, output_dir: Path, whisper_model: str,
                  delay: float = 1.0) -> str:
    vid_id = video["id"]
    title = video["title"]
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
    # Collapse the runs of underscores that punctuation leaves behind, and trim
    # the trailing one so titles ending in "." don't produce a double separator.
    safe_title = re.sub(r"_+", "_", safe_title).strip(" _") or "untitled"

    # Resume check matches on the video ID rather than the whole filename,
    # because the upload date that starts the filename is only known after the
    # fetch below. The ID is the stable part, so this still skips correctly.
    if next(output_dir.glob(f"*_{vid_id}.txt"), None) is not None:
        return f"[SKIP] {title} (already exists)"

    # Primary path: yt-dlp, the endpoint that does not get throttled.
    # It returns the upload date alongside the captions, from the same request.
    text, raw_date = transcript_via_ytdlp(vid_id, delay)
    method = "captions"

    if not text:
        text = transcript_via_api(vid_id)
        method = "captions-api"

    if not text:
        text = transcript_via_whisper(vid_id, model_size=whisper_model)
        method = "whisper"

    if text:
        upload_date = format_upload_date(raw_date)
        # Date first so a plain alphabetical file listing is chronological.
        out_file = output_dir / f"{upload_date}_{safe_title}_{vid_id}.txt"
        header = (
            f"Upload date: {upload_date}\n"
            f"Title: {title}\n"
            f"Video ID: {vid_id}\n"
            f"URL: https://www.youtube.com/watch?v={vid_id}\n"
        )
        out_file.write_text(f"{header}\n{text}\n", encoding="utf-8")
        return f"[OK:{method}] {upload_date}  {title}"
    else:
        return f"[FAIL] {title}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos")
    parser.add_argument("url", help="YouTube video, playlist, or channel URL")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers (default: 2). Raising this raises "
                             "the odds of YouTube rate-limiting your IP.")
    parser.add_argument("--delay", type=float, default=1.0, metavar="SECONDS",
                        help="Base pause before each caption request, jittered up to 2x "
                             "(default: 1.0). Raise it if you still get rate-limited.")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/Downloads"), help="Output directory (default: ~/Downloads)")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size for fallback (default: base)")
    parser.add_argument("--limit", metavar="N",
                        help="Playlists/channels only: how many videos to take, as a count "
                             "(30) or a percentage (25%%). Skips the interactive question.")
    parser.add_argument("--select", default="newest", choices=["newest", "oldest", "random"],
                        help="Which videos --limit keeps (default: newest)")
    parser.add_argument("--all", action="store_true",
                        help="Take every video on the page without asking")
    parser.add_argument("--list", action="store_true",
                        help="Count and list the videos on the page, then exit without transcribing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    videos = get_video_ids(args.url)
    if not videos:
        print("No videos found.")
        sys.exit(1)

    if args.list:
        print(f"\nFound {len(videos)} video(s):")
        for position, video in enumerate(videos, 1):
            print(f"  {position:>4}. {video['title']}")
        sys.exit(0)

    videos = scope_videos(videos, args.limit, args.select, args.all)

    print(f"\nTranscribing {len(videos)} video(s) with {args.workers} workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_video, v, output_dir, args.whisper_model, args.delay): v
            for v in videos
        }
        for future in as_completed(futures):
            print(future.result())

    print(f"\nDone. Transcripts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
