from __future__ import annotations

import argparse
from pathlib import Path

from .client import OssClient


def cmd_upload(args: argparse.Namespace) -> None:
    result = OssClient().upload_file(args.local, args.oss)
    print(f"uploaded elapsed={result.elapsed_seconds:.3f}s")


def cmd_download(args: argparse.Namespace) -> None:
    result = OssClient().download_file(args.oss, args.local)
    print(f"downloaded elapsed={result.elapsed_seconds:.3f}s")


def cmd_exists(args: argparse.Namespace) -> None:
    exists = OssClient().exists(args.oss)
    print("exists" if exists else "missing")


def cmd_delete(args: argparse.Namespace) -> None:
    result = OssClient().delete_file(args.oss, missing_ok=args.missing_ok)
    print(f"deleted elapsed={result.elapsed_seconds:.3f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oss-tools")
    sub = parser.add_subparsers(dest="command", required=True)

    upload = sub.add_parser("upload")
    upload.add_argument("local", type=Path)
    upload.add_argument("oss")
    upload.set_defaults(func=cmd_upload)

    download = sub.add_parser("download")
    download.add_argument("oss")
    download.add_argument("local", type=Path)
    download.set_defaults(func=cmd_download)

    exists = sub.add_parser("exists")
    exists.add_argument("oss")
    exists.set_defaults(func=cmd_exists)

    delete = sub.add_parser("delete")
    delete.add_argument("oss")
    delete.add_argument("--missing-ok", action="store_true")
    delete.set_defaults(func=cmd_delete)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
