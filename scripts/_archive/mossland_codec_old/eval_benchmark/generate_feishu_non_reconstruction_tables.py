from __future__ import annotations

import argparse
import csv
import html
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parent
TABLES_DIR = EVAL_ROOT / "tables"


generate_tables_mod = importlib.import_module("scripts.mossland-codec.eval_benchmark.generate_tables")


@dataclass(frozen=True)
class FeishuTable:
    name: str
    title: str
    note: str
    source_table: str
    columns: list[str]
    headers: list[str]


def format_cell(value: Any) -> str:
    if value is None or value == "":
        return "未复现"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer() and abs(number) < 10000:
        return str(int(number))
    return f"{number:.3f}"


def load_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, str]], columns: list[str], headers: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: format_cell(row.get(column, "")) for column, header in zip(columns, headers, strict=True)})


def write_markdown(rows: list[dict[str, str]], columns: list[str], headers: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(format_cell(row.get(column, "")) for column in columns) + " |\n")


def table_xml(rows: list[dict[str, str]], columns: list[str], headers: list[str]) -> str:
    col_widths = {
        "Model": 190,
        "Eval setting": 180,
        "Task / stem": 120,
    }
    widths = "".join(f'<col width="{col_widths.get(header, 105)}"/>' for header in headers)
    head = "".join(
        f'<th background-color="light-gray" vertical-align="top"><p>{html.escape(header)}</p></th>'
        for header in headers
    )
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            + "".join(
                f'<td vertical-align="top"><p>{html.escape(format_cell(row.get(column, "")))}</p></td>'
                for column in columns
            )
            + "</tr>"
        )
    return (
        "<table>\n"
        f"<colgroup>{widths}</colgroup>\n"
        f"<thead><tr>{head}</tr></thead>\n"
        "<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody></table>\n"
    )


def section_xml(table: FeishuTable, rows: list[dict[str, str]]) -> str:
    return (
        f"<h2>{html.escape(table.title)}</h2>\n"
        f'<callout emoji="ℹ️" background-color="light-blue" border-color="blue">'
        f"<p>{html.escape(table.note)}</p>"
        "</callout>\n"
        + table_xml(rows, table.columns, table.headers)
    )


def feishu_tables() -> list[FeishuTable]:
    return [
        FeishuTable(
            name="feishu_source_separation_repro",
            title="Source Separation Metrics 复现",
            note=(
                "本表基于 MUSDB18-HQ test full-track，本地统一 pipeline 重新计算 museval v4、"
                "FAD-CLAP 与 waveform/STFT diagnostics；额外训练或变体 checkpoint 只作为工程 baseline。"
            ),
            source_table="separation_fulltrack_musdb18hq.csv",
            columns=[
                "model",
                "stem",
                "count",
                "museval_sdr_mean",
                "museval_sdr_median",
                "museval_sir_mean",
                "museval_sar_mean",
                "fad_clap",
                "si_sdr_db",
                "lsd",
                "mrstft",
            ],
            headers=[
                "Model",
                "Task / stem",
                "Tracks",
                "SDR mean ↑",
                "SDR median ↑",
                "SIR mean ↑",
                "SAR mean ↑",
                "FAD_clap ↓",
                "SI-SDR ↑",
                "LSD ↓",
                "MRSTFT ↓",
            ],
        ),
        FeishuTable(
            name="feishu_super_resolution_repro",
            title="Music Bandwidth Extension / Super-Resolution Metrics 复现",
            note=(
                "本表基于 MusicCaps-HF 300 条工程评测；MusicCaps-HF 是第三方音频镜像，"
                "不是各论文官方 benchmark。A2SB 是 44.1 kHz BWE 协议，AERO 是 12 kHz -> 48 kHz 协议。"
            ),
            source_table="sr300_musiccaps_hf.csv",
            columns=[
                "model",
                "task",
                "count",
                "fad_clap",
                "fad_vggish",
                "si_sdr_db",
                "snr_db",
                "lsd",
                "lsd_lf",
                "lsd_hf",
                "mrstft",
            ],
            headers=[
                "Model",
                "Eval setting",
                "Clips",
                "FAD_clap ↓",
                "FAD ↓",
                "SI-SDR ↑",
                "SNR ↑",
                "LSD ↓",
                "LSD-LF ↓",
                "LSD-HF ↓",
                "MRSTFT ↓",
            ],
        ),
        FeishuTable(
            name="feishu_mono_to_stereo_repro",
            title="Mono-to-Stereo Metrics 复现",
            note=(
                "本表基于 MusicCaps-HF mono-to-stereo 100 条工程评测；该任务多解，"
                "SI-SDR/LSD 只是 paired-reference diagnostics，排序需结合 fold-down 保真、宽度、相关性和主观听感。"
            ),
            source_table="stereo100_musiccaps_hf.csv",
            columns=[
                "model",
                "setting",
                "count",
                "fad_clap",
                "fad_vggish",
                "si_sdr_db",
                "snr_db",
                "lsd",
                "mrstft",
                "fold_down_si_sdr_db",
                "stereo_width",
                "channel_correlation",
            ],
            headers=[
                "Model",
                "Eval setting",
                "Clips",
                "FAD_clap ↓",
                "FAD ↓",
                "SI-SDR ↑",
                "SNR ↑",
                "LSD ↓",
                "MRSTFT ↓",
                "Fold-down SI-SDR ↑",
                "Stereo width ↑",
                "Channel corr",
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Feishu-ready non-reconstruction benchmark tables.")
    parser.add_argument("--eval-root", default=str(EVAL_ROOT), help="Path to scripts/mossland-codec/eval_benchmark.")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to <eval-root>/tables.")
    args = parser.parse_args()

    root = Path(args.eval_root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_tables_mod.generate_tables(root, output_dir)

    sections = []
    for table in feishu_tables():
        rows = load_source_rows(output_dir / table.source_table)
        base = output_dir / table.name
        write_csv(rows, table.columns, table.headers, base.with_suffix(".csv"))
        write_markdown(rows, table.columns, table.headers, base.with_suffix(".md"))
        xml = section_xml(table, rows)
        base.with_suffix(".xml").write_text(xml, encoding="utf-8")
        sections.append(xml)

    combined = (
        "<h1>非重建任务 Benchmark 复现</h1>\n"
        '<callout emoji="ℹ️" background-color="light-blue" border-color="blue">'
        "<p>以下三张表只汇总本地已落盘的 source separation、super-resolution / bandwidth extension、mono-to-stereo 指标；未把 reconstruction 表重复写入。</p>"
        "</callout>\n"
        + "<hr/>\n".join(sections)
    )
    (output_dir / "feishu_non_reconstruction_repro.xml").write_text(combined, encoding="utf-8")
    print(f"wrote Feishu non-reconstruction tables to {output_dir}")


if __name__ == "__main__":
    main()
