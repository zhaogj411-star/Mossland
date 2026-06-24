from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .manifest import write_jsonl


def proxyless_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        env.pop(key, None)
    return env


def load_metadata(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_command(command: list[str], env: dict[str, str], timeout_seconds: int) -> subprocess.CompletedProcess:
    def stringify(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    try:
        return subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(part for part in (stringify(exc.stdout), stringify(exc.stderr)) if part)
        raise TimeoutError(f"command timed out after {timeout_seconds}s: {' '.join(command)}\n{output}") from exc


def download_full_audio(
    ytid: str,
    temp_dir: Path,
    env: dict[str, str],
    timeout_seconds: int,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
) -> Path:
    output_template = str(temp_dir / "%(id)s.%(ext)s")
    url = f"https://www.youtube.com/watch?v={ytid}"
    command = [
        "yt-dlp",
        "--quiet",
        "--socket-timeout",
        "20",
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        url,
    ]
    if cookies:
        command[1:1] = ["--cookies", cookies]
    if cookies_from_browser:
        command[1:1] = ["--cookies-from-browser", cookies_from_browser]
    result = run_command(command, env, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    matches = sorted(temp_dir.glob(f"{ytid}.*"))
    if not matches:
        raise FileNotFoundError(f"yt-dlp produced no file for {ytid}")
    return matches[0]


def trim_audio(
    input_path: Path,
    output_path: Path,
    start_s: float,
    duration_s: float,
    sample_rate: int,
    env: dict[str, str],
    timeout_seconds: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_s),
        "-t",
        str(duration_s),
        "-i",
        str(input_path),
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    result = run_command(command, env, timeout_seconds=timeout_seconds)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())


def download_rows(
    rows: list[dict],
    audio_dir: Path,
    sample_rate: int,
    max_items: int,
    start_index: int = 0,
    use_env_proxy: bool = False,
    timeout_seconds: int = 180,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
) -> tuple[list[dict], list[dict]]:
    env = dict(os.environ) if use_env_proxy else proxyless_env()
    successes = []
    failures = []
    attempted = 0
    for row in rows[start_index:]:
        if max_items > 0 and attempted >= max_items:
            break
        attempted += 1
        ytid = row["ytid"]
        output_path = audio_dir / f"{ytid}.wav"
        if output_path.exists() and output_path.stat().st_size > 0:
            successes.append({**row, "audio_path": str(output_path.resolve()), "status": "exists"})
            continue
        try:
            with tempfile.TemporaryDirectory(prefix=f"musiccaps-{ytid}-") as temp:
                full_audio = download_full_audio(
                    ytid,
                    Path(temp),
                    env,
                    timeout_seconds=timeout_seconds,
                    cookies=cookies,
                    cookies_from_browser=cookies_from_browser,
                )
                trim_audio(
                    full_audio,
                    output_path,
                    start_s=float(row["start_s"]),
                    duration_s=float(row["end_s"]) - float(row["start_s"]),
                    sample_rate=sample_rate,
                    env=env,
                    timeout_seconds=timeout_seconds,
                )
            successes.append({**row, "audio_path": str(output_path.resolve()), "status": "downloaded"})
            print(f"downloaded {ytid} -> {output_path}", flush=True)
        except Exception as exc:
            failures.append({**row, "status": "failed", "error": str(exc)})
            print(f"failed {ytid}: {exc}", flush=True)
    return successes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and trim MusicCaps YouTube audio clips.")
    parser.add_argument("--metadata", required=True, help="MusicCaps metadata JSONL.")
    parser.add_argument("--audio-dir", required=True, help="Output directory for clipped wav files named {ytid}.wav.")
    parser.add_argument("--success-output", default=None, help="Optional JSONL receipt for successful clips.")
    parser.add_argument("--failure-output", default=None, help="Optional JSONL receipt for failed clips.")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--use-env-proxy", action="store_true", help="Use current proxy environment instead of clearing it.")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--cookies", default=None, help="Netscape-format cookies file for yt-dlp.")
    parser.add_argument("--cookies-from-browser", default=None, help="Browser name/profile spec for yt-dlp cookies-from-browser.")
    args = parser.parse_args()

    rows = load_metadata(Path(args.metadata))
    successes, failures = download_rows(
        rows,
        audio_dir=Path(args.audio_dir),
        sample_rate=args.sample_rate,
        max_items=args.max_items,
        start_index=args.start_index,
        use_env_proxy=args.use_env_proxy,
        timeout_seconds=args.timeout_seconds,
        cookies=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
    )
    if args.success_output:
        write_jsonl(successes, args.success_output)
    if args.failure_output:
        write_jsonl(failures, args.failure_output)
    print(f"success={len(successes)} failed={len(failures)}")


if __name__ == "__main__":
    main()
