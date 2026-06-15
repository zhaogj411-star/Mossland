from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


REGISTRY = Path(__file__).with_name("public_datasets.json")


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def build_session(no_proxy: bool) -> requests.Session:
    session = requests.Session()
    if no_proxy:
        session.trust_env = False
    return session


def verify_size(path: Path, expected_size: int | None) -> bool:
    if expected_size is None:
        return path.exists() and path.stat().st_size > 0
    return path.exists() and path.stat().st_size == expected_size


def head_file(session: requests.Session, url: str) -> dict[str, str | int | None]:
    response = session.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    return {
        "status_code": response.status_code,
        "content_length": response.headers.get("content-length"),
        "accept_ranges": response.headers.get("accept-ranges"),
        "url": response.url,
    }


def download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    expected_size: int | None = None,
    chunk_size: int = 1024 * 1024,
    progress_seconds: int = 30,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verify_size(output_path, expected_size):
        print(f"exists complete {output_path}")
        return
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    downloaded = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    run_start_bytes = downloaded
    started_at = time.monotonic()
    last_report = started_at
    last_bytes = downloaded
    with session.get(url, stream=True, headers=headers, timeout=60) as response:
        response.raise_for_status()
        mode = "ab" if downloaded and response.status_code == 206 else "wb"
        if mode == "wb":
            downloaded = 0
            run_start_bytes = 0
            last_bytes = 0
        with temp_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if progress_seconds > 0 and now - last_report >= progress_seconds:
                        interval_speed = (downloaded - last_bytes) / max(now - last_report, 1e-6)
                        total_speed = (downloaded - run_start_bytes) / max(now - started_at, 1e-6)
                        eta = "unknown"
                        if expected_size and total_speed > 0:
                            eta_seconds = max(expected_size - downloaded, 0) / total_speed
                            eta = f"{eta_seconds / 60.0:.1f} min"
                        print(
                            f"progress {output_path.name}: "
                            f"{downloaded}/{expected_size or '?'} bytes "
                            f"speed={interval_speed / 1048576.0:.2f} MiB/s eta={eta}",
                            flush=True,
                        )
                        last_report = now
                        last_bytes = downloaded
    if expected_size is not None and temp_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"downloaded size mismatch for {output_path}: "
            f"{temp_path.stat().st_size} != {expected_size}"
        )
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public evaluation datasets registered for Mossland.")
    parser.add_argument("--dataset", required=True, choices=sorted(load_registry()))
    parser.add_argument("--output-dir", default="scripts/mossland-codec/eval_benchmark/data")
    parser.add_argument("--dry-run", action="store_true", help="Print files without downloading.")
    parser.add_argument("--head", action="store_true", help="Check remote headers without downloading.")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore proxy environment variables for this download.")
    parser.add_argument("--progress-seconds", type=int, default=30, help="Seconds between download progress lines.")
    args = parser.parse_args()

    registry = load_registry()
    dataset = registry[args.dataset]
    output_dir = Path(args.output_dir) / args.dataset
    session = build_session(no_proxy=args.no_proxy)
    print(f"{args.dataset}: {dataset['title']} ({dataset['doi']})")
    for file_info in dataset["files"]:
        output_path = output_dir / file_info["filename"]
        print(f"{file_info['filename']} size={file_info['size_bytes']} url={file_info['download_url']}")
        if args.head:
            print(json.dumps(head_file(session, file_info["download_url"]), ensure_ascii=False, sort_keys=True))
        if not args.dry_run and not args.head:
            download_file(
                session,
                file_info["download_url"],
                output_path,
                expected_size=file_info.get("size_bytes"),
                progress_seconds=args.progress_seconds,
            )
            print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
