from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_ROOT = Path("scripts/mossland-codec/eval_benchmark")


@dataclass(frozen=True)
class SummaryRef:
    path: str
    bucket: str


@dataclass(frozen=True)
class TableRow:
    label: str
    data: dict[str, Any]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_from_summary(root: Path, ref: SummaryRef, metric: str) -> float | None:
    summary = load_json(root / ref.path)
    if not summary:
        return None
    bucket = summary.get(ref.bucket)
    if not isinstance(bucket, dict):
        return None
    value = bucket.get(metric)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def count_from_summary(root: Path, ref: SummaryRef) -> int | None:
    value = metric_from_summary(root, ref, "count")
    if value is None:
        return None
    return int(value)


def read_visqol_summary(root: Path, path: str) -> dict[str, float] | None:
    summary = load_json(root / path / "summary.json")
    if summary and "reconstruct" in summary:
        return summary["reconstruct"]

    csv_path = root / path / "visqol_results.csv"
    if not csv_path.exists():
        return None
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("moslqo") or row.get("mos_lqo") or row.get("MOS-LQO")
            if raw:
                values.append(float(raw))
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
    return {
        "count": float(len(values)),
        "visqol_moslqo/mean": float(sum(values) / len(values)),
        "visqol_moslqo/median": float(median),
    }


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_table(rows: list[TableRow], columns: list[str], output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    with output_base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.data.get(column, "") for column in columns})

    with output_base.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(format_cell(row.data.get(column, "")) for column in columns)
                + " |\n"
            )


def codec_rows(root: Path) -> list[TableRow]:
    specs = [
        (
            "Mossland ckpt/mossland-codec0613",
            "MusicCaps-HF full",
            SummaryRef("runs/mossland_musiccaps_full_reconstruct_clap_emb/summary.json", "reconstruct"),
            SummaryRef("runs/mossland_musiccaps_full_reconstruct_vggish_emb/summary.json", "reconstruct"),
            "runs/visqol_full_codec/mossland",
        ),
        (
            "EnCodec 48 kHz 24 kbps",
            "MusicCaps-HF full",
            SummaryRef("runs/baselines_musiccaps_full/encodec_eval_clap/summary.json", "reconstruct"),
            SummaryRef("runs/baselines_musiccaps_full/encodec_eval_vggish/summary.json", "reconstruct"),
            "runs/visqol_full_codec/encodec",
        ),
        (
            "DAC 44.1 kHz 8 kbps",
            "MusicCaps-HF full",
            SummaryRef("runs/baselines_musiccaps_full/dac_eval_clap/summary.json", "reconstruct"),
            SummaryRef("runs/baselines_musiccaps_full/dac_eval_vggish/summary.json", "reconstruct"),
            "runs/visqol_full_codec/dac",
        ),
        (
            "SNAC 44 kHz",
            "MusicCaps-HF full",
            SummaryRef("runs/baselines_musiccaps_full_snac44k/eval_clap/summary.json", "reconstruct"),
            SummaryRef("runs/baselines_musiccaps_full_snac44k/eval_vggish/summary.json", "reconstruct"),
            "runs/visqol_full_codec/snac",
        ),
        (
            "WavTokenizer music 75-token",
            "MusicCaps-HF full",
            SummaryRef("runs/baselines_musiccaps_full_wavtokenizer/eval_clap/summary.json", "reconstruct"),
            SummaryRef("runs/baselines_musiccaps_full_wavtokenizer/eval_vggish/summary.json", "reconstruct"),
            "runs/visqol_full_codec/wavtokenizer",
        ),
        (
            "CoDiCodec official checkpoint",
            "MusicCaps-HF full",
            SummaryRef("runs/baselines_musiccaps_full_codicodec/eval_clap/summary.json", "reconstruct"),
            SummaryRef("runs/baselines_musiccaps_full_codicodec/eval_vggish/summary.json", "reconstruct"),
            "runs/visqol_full_codec/codicodec",
        ),
    ]
    rows: list[TableRow] = []
    for label, dataset, clap, vggish, visqol_path in specs:
        visqol = read_visqol_summary(root, visqol_path)
        count = count_from_summary(root, clap) or count_from_summary(root, vggish)
        visqol_count = int(visqol["count"]) if visqol else None
        rows.append(
            TableRow(
                label=label,
                data={
                    "model": label,
                    "dataset": dataset,
                    "count": count or "",
                    "fad_clap": metric_from_summary(root, clap, "fad_clap"),
                    "fad_vggish": metric_from_summary(root, vggish, "fad_vggish"),
                    "visqol_moslqo": visqol.get("visqol_moslqo/mean") if visqol else "",
                    "si_sdr_db": metric_from_summary(root, clap, "si_sdr_db/mean"),
                    "snr_db": metric_from_summary(root, clap, "snr_db/mean"),
                    "lsd": metric_from_summary(root, clap, "lsd/mean"),
                    "mrstft": metric_from_summary(root, clap, "mrstft/mean"),
                },
            )
        )
    return rows


def encodec_rate_rows(root: Path) -> list[TableRow]:
    specs = [
        ("EnCodec 48 kHz", "3 kbps", "runs/baselines_musiccaps_full_encodec_bw3"),
        ("EnCodec 48 kHz", "6 kbps", "runs/baselines_musiccaps_full_encodec_bw6"),
        ("EnCodec 48 kHz", "12 kbps", "runs/baselines_musiccaps_full_encodec_bw12"),
        ("EnCodec 48 kHz", "24 kbps", "runs/baselines_musiccaps_full"),
    ]
    rows = []
    for model, bandwidth, base in specs:
        clap_dir = "encodec_eval_clap" if bandwidth == "24 kbps" else "eval_clap"
        vggish_dir = "encodec_eval_vggish" if bandwidth == "24 kbps" else "eval_vggish"
        clap = SummaryRef(f"{base}/{clap_dir}/summary.json", "reconstruct")
        vggish = SummaryRef(f"{base}/{vggish_dir}/summary.json", "reconstruct")
        rows.append(
            TableRow(
                label=f"{model} {bandwidth}",
                data={
                    "model": model,
                    "bandwidth": bandwidth,
                    "count": count_from_summary(root, clap) or "",
                    "fad_clap": metric_from_summary(root, clap, "fad_clap"),
                    "fad_vggish": metric_from_summary(root, vggish, "fad_vggish"),
                    "si_sdr_db": metric_from_summary(root, clap, "si_sdr_db/mean"),
                    "snr_db": metric_from_summary(root, clap, "snr_db/mean"),
                    "lsd": metric_from_summary(root, clap, "lsd/mean"),
                    "mrstft": metric_from_summary(root, clap, "mrstft/mean"),
                },
            )
        )
    return rows


def semanticodec_rows(root: Path) -> list[TableRow]:
    rows = []
    for token_rate in (25, 50, 100):
        base = f"runs/baselines_musiccaps_semanticodec100_tr{token_rate}"
        clap = SummaryRef(f"{base}/eval_clap/summary.json", "reconstruct")
        vggish = SummaryRef(f"{base}/eval_vggish/summary.json", "reconstruct")
        rows.append(
            TableRow(
                label=f"SemantiCodec {token_rate}",
                data={
                    "model": "SemantiCodec official",
                    "token_rate": token_rate,
                    "count": count_from_summary(root, clap) or "",
                    "fad_clap": metric_from_summary(root, clap, "fad_clap"),
                    "fad_vggish": metric_from_summary(root, vggish, "fad_vggish"),
                    "si_sdr_db": metric_from_summary(root, clap, "si_sdr_db/mean"),
                    "snr_db": metric_from_summary(root, clap, "snr_db/mean"),
                    "lsd": metric_from_summary(root, clap, "lsd/mean"),
                    "mrstft": metric_from_summary(root, clap, "mrstft/mean"),
                },
            )
        )
    return rows


def sr300_rows(root: Path) -> list[TableRow]:
    specs = [
        ("Mossland ckpt/mossland-codec0613", "16 kHz -> 48 kHz", "16000", "runs/mossland_musiccaps_sr300_16000", "_flat"),
        ("Mossland ckpt/mossland-codec0613", "24 kHz -> 48 kHz", "24000", "runs/mossland_musiccaps_sr300_24000", "_flat"),
        ("Mossland ckpt/mossland-codec0613", "32 kHz -> 48 kHz", "32000", "runs/mossland_musiccaps_sr300_32000", "_flat"),
        ("AudioSR basic", "16 kHz -> 48 kHz", "16000", "runs/baselines_musiccaps_sr300_16000_audiosr", ""),
        ("AudioSR basic", "24 kHz -> 48 kHz", "24000", "runs/baselines_musiccaps_sr300_24000_audiosr", ""),
        ("AudioSR basic", "32 kHz -> 48 kHz", "32000", "runs/baselines_musiccaps_sr300_32000_audiosr", ""),
        ("FlowHigh", "16 kHz -> 48 kHz", "16000", "runs/baselines_musiccaps_sr300_16000_flowhigh", ""),
        ("FlowHigh", "24 kHz -> 48 kHz", "24000", "runs/baselines_musiccaps_sr300_24000_flowhigh", ""),
        ("FlowHigh", "32 kHz -> 48 kHz", "32000", "runs/baselines_musiccaps_sr300_32000_flowhigh", ""),
        ("NU-Wave2", "16 kHz -> 48 kHz", "16000", "runs/baselines_musiccaps_sr300_16000_nuwave2", ""),
        ("NU-Wave2", "24 kHz -> 48 kHz", "24000", "runs/baselines_musiccaps_sr300_24000_nuwave2", ""),
        ("NU-Wave2", "32 kHz -> 48 kHz", "32000", "runs/baselines_musiccaps_sr300_32000_nuwave2", ""),
        ("FastWave official checkpoint", "16 kHz -> 48 kHz", "16000", "runs/baselines_musiccaps_sr300_16000_fastwave", ""),
        ("FastWave official checkpoint", "24 kHz -> 48 kHz", "24000", "runs/baselines_musiccaps_sr300_24000_fastwave", ""),
        ("FastWave official checkpoint", "32 kHz -> 48 kHz", "32000", "runs/baselines_musiccaps_sr300_32000_fastwave", ""),
        ("UniverSR audio", "16 kHz -> 48 kHz", "16000", "runs/baselines_musiccaps_sr300_16000_universr", ""),
        ("UniverSR audio", "24 kHz -> 48 kHz", "24000", "runs/baselines_musiccaps_sr300_24000_universr", ""),
        ("AERO official hl=128", "12 kHz -> 48 kHz", "12000", "runs/baselines_musiccaps_sr300_12000_aero", ""),
        ("AERO official hl=256", "12 kHz -> 48 kHz", "12000", "runs/baselines_musiccaps_sr300_12000_aero_hl256", ""),
        ("A2SB official", "16 kHz low-band -> 44.1 kHz BWE", "16000", "runs/baselines_musiccaps_sr300_16000_a2sb", ""),
    ]
    rows = []
    for model, task, rate, base, layout in specs:
        bucket = f"super_resolution@{rate}"
        if layout == "_flat":
            clap = SummaryRef(f"{base}_clap/summary.json", bucket)
            vggish = SummaryRef(f"{base}_vggish/summary.json", bucket)
        else:
            clap = SummaryRef(f"{base}/eval_clap/summary.json", bucket)
            vggish = SummaryRef(f"{base}/eval_vggish/summary.json", bucket)
        rows.append(
            TableRow(
                label=f"{model} {task}",
                data={
                    "model": model,
                    "task": task,
                    "count": count_from_summary(root, clap) or "",
                    "fad_clap": metric_from_summary(root, clap, "fad_clap"),
                    "fad_vggish": metric_from_summary(root, vggish, "fad_vggish"),
                    "si_sdr_db": metric_from_summary(root, clap, "si_sdr_db/mean"),
                    "snr_db": metric_from_summary(root, clap, "snr_db/mean"),
                    "lsd": metric_from_summary(root, clap, "lsd/mean"),
                    "lsd_lf": metric_from_summary(root, clap, "lsd_lf/mean"),
                    "lsd_hf": metric_from_summary(root, clap, "lsd_hf/mean"),
                    "mrstft": metric_from_summary(root, clap, "mrstft/mean"),
                },
            )
        )
    return rows


def separation_fulltrack_rows(root: Path) -> list[TableRow]:
    specs = [
        ("Mossland chunked full-track", "runs/mossland_musdb18hq_test_fulltrack_chunked", "runs/mossland_musdb18hq_test_fulltrack_chunked_museval"),
        ("HTDemucs two-stem", "runs/baselines_musdb18hq_fulltrack_demucs", None),
        ("HTDemucs-MMI", "runs/baselines_musdb18hq_fulltrack_hdemucs_mmi", None),
        ("HTDemucs-FT two-stem", "runs/baselines_musdb18hq_fulltrack_htdemucs_ft", None),
        ("Demucs MDX", "runs/baselines_musdb18hq_fulltrack_demucs_mdx", None),
        ("Demucs MDX Extra", "runs/baselines_musdb18hq_fulltrack_mdx_extra", None),
        ("Open-Unmix UMXHQ", "runs/baselines_musdb18hq_fulltrack_openunmix", None),
        ("Open-Unmix UMXL", "runs/baselines_musdb18hq_fulltrack_openunmix_umxl", None),
        ("BS-RoFormer v1.0.12", "runs/baselines_musdb18hq_fulltrack_bs_roformer", None),
        ("SCNet XL IHF v1.0.15", "runs/baselines_musdb18hq_fulltrack_scnet_xl_ihf", None),
        ("ByteSep ResUNet143", "runs/baselines_musdb18hq_fulltrack_bytesep_resunet", None),
    ]
    rows = []
    for model, base, museval_base in specs:
        museval_dir = museval_base or f"{base}/museval"
        for stem, bucket in (("vocals", "separate_vocals"), ("accompaniment", "separate_accompaniment")):
            museval = SummaryRef(f"{museval_dir}/summary.json", stem)
            none = SummaryRef(f"{base}/eval_none/summary.json", bucket)
            clap = SummaryRef(f"{base}/eval_clap/summary.json", bucket)
            rows.append(
                TableRow(
                    label=f"{model} {stem}",
                    data={
                        "model": model,
                        "stem": stem,
                        "count": count_from_summary(root, none) or count_from_summary(root, museval) or "",
                        "museval_sdr_mean": metric_from_summary(root, museval, "museval_v4_sdr_db/mean"),
                        "museval_sdr_median": metric_from_summary(root, museval, "museval_v4_sdr_db/median"),
                        "museval_sir_mean": metric_from_summary(root, museval, "museval_v4_sir_db/mean"),
                        "museval_sar_mean": metric_from_summary(root, museval, "museval_v4_sar_db/mean"),
                        "fad_clap": metric_from_summary(root, clap, "fad_clap"),
                        "si_sdr_db": metric_from_summary(root, none, "si_sdr_db/mean"),
                        "snr_db": metric_from_summary(root, none, "snr_db/mean"),
                        "lsd": metric_from_summary(root, none, "lsd/mean"),
                        "mrstft": metric_from_summary(root, none, "mrstft/mean"),
                    },
                )
            )
    return rows


def stereo100_rows(root: Path) -> list[TableRow]:
    specs = [
        (
            "Mossland ckpt/mossland-codec0613",
            "MusicCaps-HF seed0+seed1",
            "runs/mossland_musiccaps_multitask100_clap/summary.json",
            "runs/mossland_musiccaps_multitask100_vggish/summary.json",
        ),
        (
            "DiffStereo official sampler seed0",
            "MusicCaps-HF seed0",
            "runs/baselines_musiccaps_stereo100_official/eval_clap/summary.json",
            "runs/baselines_musiccaps_stereo100_official/eval_vggish/summary.json",
        ),
        (
            "DiffStereo official sampler seed1",
            "MusicCaps-HF seed1",
            "runs/baselines_musiccaps_stereo100_seed1_official/eval_clap/summary.json",
            "runs/baselines_musiccaps_stereo100_seed1_official/eval_vggish/summary.json",
        ),
        (
            "Ambisonizer official checkpoint",
            "MusicCaps-HF seed0",
            "runs/baselines_musiccaps_stereo100_ambisonizer/eval_clap/summary.json",
            "runs/baselines_musiccaps_stereo100_ambisonizer/eval_vggish/summary.json",
        ),
        (
            "s3a fallback decorrelator",
            "MusicCaps-HF seed0",
            "runs/baselines_musiccaps_stereo100_s3a/eval_clap/summary.json",
            "runs/baselines_musiccaps_stereo100_s3a/eval_vggish/summary.json",
        ),
    ]
    rows = []
    for model, setting, clap_path, vggish_path in specs:
        clap = SummaryRef(clap_path, "mono_to_stereo")
        vggish = SummaryRef(vggish_path, "mono_to_stereo")
        rows.append(
            TableRow(
                label=model,
                data={
                    "model": model,
                    "setting": setting,
                    "count": count_from_summary(root, clap) or "",
                    "fad_clap": metric_from_summary(root, clap, "fad_clap"),
                    "fad_vggish": metric_from_summary(root, vggish, "fad_vggish"),
                    "si_sdr_db": metric_from_summary(root, clap, "si_sdr_db/mean"),
                    "snr_db": metric_from_summary(root, clap, "snr_db/mean"),
                    "lsd": metric_from_summary(root, clap, "lsd/mean"),
                    "mrstft": metric_from_summary(root, clap, "mrstft/mean"),
                    "fold_down_si_sdr_db": metric_from_summary(root, clap, "fold_down_si_sdr_db/mean"),
                    "stereo_width": metric_from_summary(root, clap, "stereo_width/mean"),
                    "channel_correlation": metric_from_summary(root, clap, "channel_correlation/mean"),
                },
            )
        )
    return rows


def generate_tables(root: Path, output_dir: Path) -> None:
    tables = {
        "codec_full_musiccaps_hf": (
            codec_rows(root),
            [
                "model",
                "dataset",
                "count",
                "fad_clap",
                "fad_vggish",
                "visqol_moslqo",
                "si_sdr_db",
                "snr_db",
                "lsd",
                "mrstft",
            ],
        ),
        "encodec_rate_quality_full_musiccaps_hf": (
            encodec_rate_rows(root),
            ["model", "bandwidth", "count", "fad_clap", "fad_vggish", "si_sdr_db", "snr_db", "lsd", "mrstft"],
        ),
        "semanticodec_100_musiccaps_hf": (
            semanticodec_rows(root),
            ["model", "token_rate", "count", "fad_clap", "fad_vggish", "si_sdr_db", "snr_db", "lsd", "mrstft"],
        ),
        "sr300_musiccaps_hf": (
            sr300_rows(root),
            [
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
        ),
        "separation_fulltrack_musdb18hq": (
            separation_fulltrack_rows(root),
            [
                "model",
                "stem",
                "count",
                "museval_sdr_mean",
                "museval_sdr_median",
                "museval_sir_mean",
                "museval_sar_mean",
                "fad_clap",
                "si_sdr_db",
                "snr_db",
                "lsd",
                "mrstft",
            ],
        ),
        "stereo100_musiccaps_hf": (
            stereo100_rows(root),
            [
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
        ),
    }
    for name, (rows, columns) in tables.items():
        write_table(rows, columns, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark reproduction tables from local eval summaries.")
    parser.add_argument("--eval-root", default=str(EVAL_ROOT), help="Path to scripts/mossland-codec/eval_benchmark.")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to <eval-root>/tables.")
    args = parser.parse_args()
    root = Path(args.eval_root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "tables"
    generate_tables(root, output_dir)
    print(f"wrote tables to {output_dir}")


if __name__ == "__main__":
    main()
