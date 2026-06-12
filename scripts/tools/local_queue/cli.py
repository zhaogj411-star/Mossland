from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .devices import detect_devices
from .finalize import finalize
from .manifest import write_manifest
from .monitor import snapshot
from .registry import load_handler
from .runner import run_worker


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.root)
    handler = load_handler(args.handler)
    config = load_config(args.config)
    tasks = handler.build_tasks(root, args.job, config)
    count = write_manifest(
        root,
        args.job,
        tasks,
        {
            "job": args.job,
            "handler": args.handler,
            "task_type": handler.task_type,
            "config": config,
        },
    )
    print(f"created manifest for job={args.job} tasks={count}")


def cmd_worker(args: argparse.Namespace) -> None:
    handler = load_handler(args.handler)
    run_worker(
        root=Path(args.root),
        job=args.job,
        handler=handler,
        device=args.device,
        lease_ttl=args.lease_ttl,
        heartbeat_interval=args.heartbeat_interval,
        max_attempts=args.max_attempts,
        once=args.once,
    )


def cmd_start(args: argparse.Namespace) -> None:
    devices = detect_devices() if args.devices == "auto" else [x.strip() for x in args.devices.split(",") if x.strip()]
    procs: list[subprocess.Popen] = []

    for device in devices:
        cmd = [
            sys.executable,
            "-m",
            "local_queue.cli",
            "worker",
            "--root",
            args.root,
            "--job",
            args.job,
            "--handler",
            args.handler,
            "--device",
            device,
            "--lease-ttl",
            str(args.lease_ttl),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-attempts",
            str(args.max_attempts),
        ]
        procs.append(subprocess.Popen(cmd))

    while True:
        for i, proc in enumerate(procs):
            if proc.poll() is not None:
                device = devices[i]
                cmd = [
                    sys.executable,
                    "-m",
                    "local_queue.cli",
                    "worker",
                    "--root",
                    args.root,
                    "--job",
                    args.job,
                    "--handler",
                    args.handler,
                    "--device",
                    device,
                    "--lease-ttl",
                    str(args.lease_ttl),
                    "--heartbeat-interval",
                    str(args.heartbeat_interval),
                    "--max-attempts",
                    str(args.max_attempts),
                ]
                procs[i] = subprocess.Popen(cmd)
        time.sleep(10)


def cmd_monitor(args: argparse.Namespace) -> None:
    info = snapshot(Path(args.root), args.job)
    for key, value in info.items():
        print(f"{key}: {value}")


def cmd_finalize(args: argparse.Namespace) -> None:
    info = finalize(Path(args.root), args.job, allow_failed=args.allow_failed)
    for key in ["job", "total", "done", "failed", "missing", "success"]:
        print(f"{key}: {info[key]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-queue")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--job", required=True)
    init.add_argument("--handler", required=True)
    init.add_argument("--config")
    init.set_defaults(func=cmd_init)

    worker = sub.add_parser("worker")
    worker.add_argument("--root", required=True)
    worker.add_argument("--job", required=True)
    worker.add_argument("--handler", required=True)
    worker.add_argument("--device", default="cpu")
    worker.add_argument("--lease-ttl", type=int, default=300)
    worker.add_argument("--heartbeat-interval", type=int, default=30)
    worker.add_argument("--max-attempts", type=int, default=5)
    worker.add_argument("--once", action="store_true")
    worker.set_defaults(func=cmd_worker)

    start = sub.add_parser("start")
    start.add_argument("--root", required=True)
    start.add_argument("--job", required=True)
    start.add_argument("--handler", required=True)
    start.add_argument("--devices", default="auto")
    start.add_argument("--lease-ttl", type=int, default=300)
    start.add_argument("--heartbeat-interval", type=int, default=30)
    start.add_argument("--max-attempts", type=int, default=5)
    start.set_defaults(func=cmd_start)

    monitor = sub.add_parser("monitor")
    monitor.add_argument("--root", required=True)
    monitor.add_argument("--job", required=True)
    monitor.set_defaults(func=cmd_monitor)

    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--root", required=True)
    finalize_parser.add_argument("--job", required=True)
    finalize_parser.add_argument("--allow-failed", action="store_true")
    finalize_parser.set_defaults(func=cmd_finalize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
