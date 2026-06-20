from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path


TABLES_DIR = Path(__file__).resolve().parent / "tables"


@dataclass(frozen=True)
class TableSpec:
    name: str
    headers: list[str]
    rows: list[list[str]]
    widths: list[int]


def cell(value: str) -> str:
    return value if value else "未复现"


def write_xml(spec: TableSpec) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    colgroup = "".join(f'<col width="{width}"/>' for width in spec.widths)
    header = "".join(
        f'<th background-color="light-gray" vertical-align="top"><p>{html.escape(name)}</p></th>'
        for name in spec.headers
    )
    body_rows = []
    for row in spec.rows:
        body_rows.append(
            "<tr>"
            + "".join(
                f'<td vertical-align="top"><p>{html.escape(cell(value))}</p></td>' for value in row
            )
            + "</tr>"
        )
    content = (
        "<table>\n"
        f"<colgroup>{colgroup}</colgroup>\n"
        f"<thead><tr>{header}</tr></thead>\n"
        "<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody></table>\n"
    )
    (TABLES_DIR / f"{spec.name}.xml").write_text(content, encoding="utf-8")


def specs() -> list[TableSpec]:
    separation_headers = [
        "Model",
        "Checkpoint / Commit",
        "Data source",
        "Eval split",
        "Vocals SDR ↑",
        "Drums SDR ↑",
        "Bass SDR ↑",
        "Other SDR ↑",
        "Avg SDR ↑",
        "Vocals SI-SDR ↑",
        "Drums SI-SDR ↑",
        "Bass SI-SDR ↑",
        "Other SI-SDR ↑",
        "Avg SI-SDR ↑",
        "Params ↓",
        "RTF / Latency ↓",
        "Status",
    ]
    sr_headers = [
        "Model",
        "Checkpoint / Commit",
        "Input cutoff",
        "Target SR / bandwidth",
        "Eval dataset",
        "LSD ↓",
        "LSD-HF ↓",
        "STFT distance ↓",
        "SI-SDR ↑",
        "ViSQOL / MOS ↑",
        "FD / FAD ↓",
        "RTF ↓",
        "Status",
    ]
    stereo_headers = [
        "Model",
        "Checkpoint / Commit",
        "Eval dataset",
        "Mid LSD ↓",
        "Side LSD ↓",
        "Left LSD ↓",
        "Right LSD ↓",
        "Mid SI-SDR / SI-SNR ↑",
        "Side SI-SDR / SI-SNR ↑",
        "Left SI-SDR / SI-SNR ↑",
        "Right SI-SDR / SI-SNR ↑",
        "Width → ref",
        "Channel corr → ref",
        "Phase corr → ref",
        "FAD ↓",
        "Status",
    ]
    return [
        TableSpec(
            name="feishu_existing_source_separation_repro",
            headers=separation_headers,
            widths=[82, 170, 115, 110, 78, 78, 78, 78, 78, 86, 86, 86, 86, 86, 74, 96, 180],
            rows=[
                [
                    "SCNet",
                    "MSST SCNet XL IHF v1.0.15; model_scnet_ep_36_sdr_10.0891.ckpt",
                    "MUSDB18-HQ test; official public checkpoint",
                    "full-track 4-stem eval",
                    "11.422",
                    "11.816",
                    "9.236",
                    "7.883",
                    "10.089",
                    "10.699",
                    "11.523",
                    "7.408",
                    "6.797",
                    "9.106",
                    "",
                    "",
                    "已复现 4-stem；SDR 为 waveform energy ratio，SI-SDR 为 scale-invariant",
                ],
                [
                    "Mel-RoFormer",
                    "MSST Mel-RoFormer v1.0.11; model_mel_band_roformer_ep_5_sdr_8.9443.ckpt",
                    "MUSDB18-HQ test; official public checkpoint",
                    "full-track 4-stem debug run",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "未复现：checkpoint/config 已下载并跑完，但官方 inference raw outputs 全零，指标不计入",
                ],
                [
                    "BS-RoFormer",
                    "MSST BS-RoFormer v1.0.12; model_bs_roformer_ep_17_sdr_9.6568.ckpt",
                    "MUSDB18-HQ test; MUSDB18HQ-only checkpoint",
                    "full-track 4-stem eval",
                    "11.008",
                    "11.552",
                    "8.438",
                    "7.439",
                    "9.609",
                    "10.336",
                    "11.229",
                    "7.562",
                    "6.265",
                    "8.848",
                    "",
                    "",
                    "已复现 4-stem；SDR 为 waveform energy ratio，SI-SDR 为 scale-invariant",
                ],
                [
                    "Band-SCNet",
                    "",
                    "MUSDB18-HQ only",
                    "MUSDB18-HQ test",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "未复现：当前未找到匹配原表的 Band-SCNet 4-stem run",
                ],
                [
                    "HT Demucs",
                    "demucs official htdemucs",
                    "MUSDB HQ plus extra songs",
                    "MUSDB18-HQ test full-track two-stem eval",
                    "6.029",
                    "",
                    "",
                    "",
                    "",
                    "7.857",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "已复现 two-stem vocals/accompaniment；原表 4-stem drums/bass/other 未复现",
                ],
                [
                    "Mossland codec 370k (EMA)",
                    "logs/mossland-codec/runs/2026-06-12_12-46-36 last.ckpt EMA export",
                    "MUSDB18-HQ test; Mossland codec checkpoint",
                    "full-track two-stem chunked eval",
                    "-0.571",
                    "",
                    "",
                    "",
                    "-0.729",
                    "-13.293",
                    "",
                    "",
                    "",
                    "-12.973",
                    "",
                    "",
                    "已复现 vocals/accompaniment full-track；drums/bass/other 不在 5-task codec embedding 中",
                ],
            ],
        ),
        TableSpec(
            name="feishu_existing_super_resolution_repro",
            headers=sr_headers,
            widths=[105, 160, 120, 125, 140, 80, 88, 90, 82, 96, 82, 70, 180],
            rows=[
                [
                    "AudioSR",
                    "haoheliu/audiosr_basic; default 50 DDIM steps, guidance 3.5",
                    "16 / 24 / 32 kHz",
                    "48 kHz",
                    "MusicCaps-HF 100 clips per bucket",
                    "31.041 / 31.415 / 31.366",
                    "37.699 / 39.569 / 39.156",
                    "0.695 / 0.681 / 0.681",
                    "11.233 / 11.423 / 11.443",
                    "2.273 / 2.216 / 2.227",
                    "2.169 / 1.936 / 1.925",
                    "",
                    "已复现；数据源为 MusicCaps-HF 第三方镜像",
                ],
                [
                    "FlashSR",
                    "ysharma3501/FlashSR official source; YatharthS/FlashSR upsampler.pth",
                    "16 kHz",
                    "48 kHz",
                    "MusicCaps-HF 100 clips",
                    "40.799",
                    "50.587",
                    "1.103",
                    "12.755",
                    "1.771",
                    "1.302",
                    "",
                    "已复现 16 kHz bucket；ViSQOL 为 Google ViSQOL audio mode",
                ],
                [
                    "Mossland codec 370k (EMA)",
                    "logs/mossland-codec/runs/2026-06-12_12-46-36 last.ckpt EMA export",
                    "16 / 24 / 32 kHz",
                    "44.1 kHz",
                    "MusicCaps-HF 300 clips per bucket",
                    "26.466 / 28.284 / 28.346",
                    "31.993 / 36.101 / 36.023",
                    "0.500 / 0.521 / 0.520",
                    "-9.507 / -9.592 / -9.574",
                    "3.231 / 2.947 / 2.922",
                    "0.590 / 0.608 / 0.612",
                    "",
                    "已复现 strict 10s；EncoderDecoder encode/decode parallel super_resolution，FAD 为 VGGish FAD，数据源为 MusicCaps-HF 第三方镜像",
                ],
                [
                    "SAGA-SR",
                    "",
                    "4 kHz / 8 kHz / custom",
                    "44.1 kHz",
                    "FMA-small / internal music set",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "未复现：当前无匹配 checkpoint / adapter run",
                ],
            ],
        ),
        TableSpec(
            name="feishu_existing_mono_to_stereo_repro",
            headers=stereo_headers,
            widths=[120, 160, 130, 70, 70, 70, 70, 85, 85, 85, 85, 80, 90, 85, 75, 175],
            rows=[
                [
                    "DiffStereo",
                    "official model_epoch_80000.pt; 1000-step sampler",
                    "MusicCaps-HF 100 seed0 / 100 seed1",
                    "",
                    "",
                    "31.863 / 31.799",
                    "31.863 / 31.799",
                    "",
                    "",
                    "0.806 / 0.964",
                    "0.806 / 0.964",
                    "0.455 / 0.439",
                    "0.651 / 0.671",
                    "",
                    "6.057 / 6.006",
                    "已复现可用指标；原表 mid/side/phase 专项指标未实现",
                ],
                [
                    "Parametric stereo baseline",
                    "s3a fallback decorrelator anchor",
                    "MusicCaps-HF 100 seed0",
                    "",
                    "",
                    "6.156",
                    "6.156",
                    "",
                    "",
                    "6.840",
                    "6.840",
                    "0.324",
                    "0.954",
                    "",
                    "0.026",
                    "已复现 DSP fallback；不是 s3a 官方 API 路径",
                ],
                [
                    "Mossland codec 370k (EMA)",
                    "logs/mossland-codec/runs/2026-06-12_12-46-36 last.ckpt EMA export",
                    "MusicCaps-HF 100 seed0 / 100 seed1",
                    "",
                    "",
                    "28.574",
                    "28.574",
                    "-9.633",
                    "",
                    "-10.485",
                    "-10.485",
                    "0.121",
                    "0.949",
                    "",
                    "3.051",
                    "已复现；当前 pipeline 无 mid/side/phase 专项，left/right 列填整体 stereo 指标",
                ],
            ],
        ),
    ]


def main() -> None:
    for spec in specs():
        write_xml(spec)
    print(f"wrote {len(specs())} existing Feishu task tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
