from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parent
TABLES_DIR = EVAL_ROOT / "tables"


@dataclass(frozen=True)
class MetricRef:
    summary_path: str
    bucket: str = "reconstruct"


@dataclass(frozen=True)
class ReproRow:
    model: str
    source: str
    stereo: str
    representation: str
    compression_ratio: str
    bitrate: str
    bandwidth_khz: str
    si_sdr: str
    visqol: str
    mel_distance: str
    stft_distance: str
    fad_clap: str
    fad: str

    def as_dict(self) -> dict[str, str]:
        return {
            "Model": self.model,
            "Source / Eval setting": self.source,
            "Stereo": self.stereo,
            "Representation": self.representation,
            "Compression Ratio": self.compression_ratio,
            "Bitrate": self.bitrate,
            "Bandwidth (kHz)": self.bandwidth_khz,
            "SI-SDR ↑": self.si_sdr,
            "ViSQOL ↑": self.visqol,
            "Mel distance ↓": self.mel_distance,
            "STFT distance ↓": self.stft_distance,
            "FAD_clap ↓": self.fad_clap,
            "FAD ↓": self.fad,
        }


HEADERS = [
    "Model",
    "Source / Eval setting",
    "Stereo",
    "Representation",
    "Compression Ratio",
    "Bitrate",
    "Bandwidth (kHz)",
    "SI-SDR ↑",
    "ViSQOL ↑",
    "Mel distance ↓",
    "STFT distance ↓",
    "FAD_clap ↓",
    "FAD ↓",
]


def load_summary(root: Path, path: str) -> dict[str, Any] | None:
    summary_path = root / path
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(root: Path, ref: MetricRef, key: str) -> float | None:
    summary = load_summary(root, ref.summary_path)
    if not summary:
        return None
    value = summary.get(ref.bucket, {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "未复现"
    return f"{value:.{digits}f}"


def distance_cols(root: Path, ref: MetricRef) -> tuple[str, str]:
    return (
        fmt(metric(root, ref, "lsd/mean")),
        fmt(metric(root, ref, "mrstft/mean")),
    )


def build_rows(root: Path = EVAL_ROOT) -> list[ReproRow]:
    source = "MusicCaps-HF full (5355/5521)"
    not_run = "未复现"

    dac_clap = MetricRef("runs/baselines_musiccaps_full/dac_eval_clap/summary.json")
    dac_vggish = MetricRef("runs/baselines_musiccaps_full/dac_eval_vggish/summary.json")
    dac_visqol = MetricRef("runs/visqol_full_codec/dac/summary.json")
    dac_rates = {
        "1.78 kbps": (
            MetricRef("runs/baselines_musiccaps_full_dac_nq2/eval_clap/summary.json"),
            MetricRef("runs/baselines_musiccaps_full_dac_nq2/eval_vggish/summary.json"),
            MetricRef("runs/visqol_full_codec/dac_nq2/summary.json"),
            "DAC n_quantizers=2",
        ),
        "2.67 kbps": (
            MetricRef("runs/baselines_musiccaps_full_dac_nq3/eval_clap/summary.json"),
            MetricRef("runs/baselines_musiccaps_full_dac_nq3/eval_vggish/summary.json"),
            MetricRef("runs/visqol_full_codec/dac_nq3/summary.json"),
            "DAC n_quantizers=3",
        ),
        "5.33 kbps": (
            MetricRef("runs/baselines_musiccaps_full_dac_nq6/eval_clap/summary.json"),
            MetricRef("runs/baselines_musiccaps_full_dac_nq6/eval_vggish/summary.json"),
            MetricRef("runs/visqol_full_codec/dac_nq6/summary.json"),
            "DAC n_quantizers=6",
        ),
        "8 kbps": (
            dac_clap,
            dac_vggish,
            dac_visqol,
            "DAC n_quantizers=9",
        ),
    }
    encodec_claps = {
        "3 kbps": MetricRef("runs/baselines_musiccaps_full_encodec_bw3/eval_clap/summary.json"),
        "6 kbps": MetricRef("runs/baselines_musiccaps_full_encodec_bw6/eval_clap/summary.json"),
        "12 kbps": MetricRef("runs/baselines_musiccaps_full_encodec_bw12/eval_clap/summary.json"),
        "24 kbps": MetricRef("runs/baselines_musiccaps_full/encodec_eval_clap/summary.json"),
    }
    encodec_vggish = {
        "3 kbps": MetricRef("runs/baselines_musiccaps_full_encodec_bw3/eval_vggish/summary.json"),
        "6 kbps": MetricRef("runs/baselines_musiccaps_full_encodec_bw6/eval_vggish/summary.json"),
        "12 kbps": MetricRef("runs/baselines_musiccaps_full_encodec_bw12/eval_vggish/summary.json"),
        "24 kbps": MetricRef("runs/baselines_musiccaps_full/encodec_eval_vggish/summary.json"),
    }
    encodec_visqol = {
        "3 kbps": MetricRef("runs/visqol_full_codec/encodec_bw3/summary.json"),
        "6 kbps": MetricRef("runs/visqol_full_codec/encodec_bw6/summary.json"),
        "12 kbps": MetricRef("runs/visqol_full_codec/encodec_bw12/summary.json"),
        "24 kbps": MetricRef("runs/visqol_full_codec/encodec/summary.json"),
    }
    codicodec_clap = MetricRef("runs/baselines_musiccaps_full_codicodec/eval_clap/summary.json")
    codicodec_vggish = MetricRef("runs/baselines_musiccaps_full_codicodec/eval_vggish/summary.json")
    codicodec_visqol = MetricRef("runs/visqol_full_codec/codicodec/summary.json")
    music2latent_clap = MetricRef("runs/baselines_musiccaps_full_music2latent/eval_clap/summary.json")
    music2latent_vggish = MetricRef("runs/baselines_musiccaps_full_music2latent/eval_vggish/summary.json")
    music2latent_visqol = MetricRef("runs/visqol_full_codec/music2latent/summary.json")
    stable_audio3_same_s_clap = MetricRef("runs/baselines_musiccaps_full_stable_audio3_same_s/eval_clap/summary.json")
    stable_audio3_same_s_vggish = MetricRef("runs/baselines_musiccaps_full_stable_audio3_same_s/eval_vggish/summary.json")
    stable_audio3_same_s_visqol = MetricRef("runs/visqol_full_codec/stable_audio3_same_s/summary.json")
    stable_audio3_same_l_clap = MetricRef("runs/baselines_musiccaps_full_stable_audio3_same_l/eval_clap/summary.json")
    stable_audio3_same_l_vggish = MetricRef("runs/baselines_musiccaps_full_stable_audio3_same_l/eval_vggish/summary.json")
    stable_audio3_same_l_visqol = MetricRef("runs/visqol_full_codec/stable_audio3_same_l/summary.json")
    codicodec_repro_10k_raw_clap = MetricRef("runs/codicodec_paper_repro_10000_raw_fullclip_xfade/eval_clap/summary.json")
    codicodec_repro_10k_raw_vggish = MetricRef("runs/codicodec_paper_repro_10000_raw_fullclip_xfade/eval_vggish/summary.json")
    codicodec_repro_10k_raw_visqol = MetricRef("runs/codicodec_paper_repro_10000_raw_fullclip_xfade/visqol/summary.json")
    codicodec_repro_10k_ema_clap = MetricRef("runs/codicodec_paper_repro_10000_ema_fullclip_xfade/eval_clap/summary.json")
    codicodec_repro_10k_ema_vggish = MetricRef("runs/codicodec_paper_repro_10000_ema_fullclip_xfade/eval_vggish/summary.json")
    codicodec_repro_10k_ema_visqol = MetricRef("runs/codicodec_paper_repro_10000_ema_fullclip_xfade/visqol/summary.json")
    codicodec_repro_18k_raw_clap = MetricRef("runs/codicodec_paper_repro_18000_raw_fullclip_xfade/eval_clap/summary.json")
    codicodec_repro_18k_raw_vggish = MetricRef("runs/codicodec_paper_repro_18000_raw_fullclip_xfade/eval_vggish/summary.json")
    codicodec_repro_18k_raw_visqol = MetricRef("runs/codicodec_paper_repro_18000_raw_fullclip_xfade/visqol/summary.json")
    codicodec_repro_18k_ema_clap = MetricRef("runs/codicodec_paper_repro_18000_ema_fullclip_xfade/eval_clap/summary.json")
    codicodec_repro_18k_ema_vggish = MetricRef("runs/codicodec_paper_repro_18000_ema_fullclip_xfade/eval_vggish/summary.json")
    codicodec_repro_18k_ema_visqol = MetricRef("runs/codicodec_paper_repro_18000_ema_fullclip_xfade/visqol/summary.json")
    mossland_codec_370k_ema_clap = MetricRef("runs/mossland_codec_370000_ema_parallel_decode/eval_clap/summary.json")
    mossland_codec_370k_ema_vggish = MetricRef("runs/mossland_codec_370000_ema_parallel_decode/eval_vggish/summary.json")
    mossland_codec_370k_ema_visqol = MetricRef("runs/mossland_codec_370000_ema_parallel_decode/visqol/summary.json")
    opus_rates = {
        "8 kbps": (
            MetricRef("runs/baselines_musiccaps_full_opus_8kbps/eval_clap/summary.json"),
            MetricRef("runs/baselines_musiccaps_full_opus_8kbps/eval_vggish/summary.json"),
            MetricRef("runs/visqol_full_codec/opus_8kbps/summary.json"),
        ),
        "14 kbps": (
            MetricRef("runs/baselines_musiccaps_full_opus_14kbps/eval_clap/summary.json"),
            MetricRef("runs/baselines_musiccaps_full_opus_14kbps/eval_vggish/summary.json"),
            MetricRef("runs/visqol_full_codec/opus_14kbps/summary.json"),
        ),
        "24 kbps": (
            MetricRef("runs/baselines_musiccaps_full_opus_24kbps/eval_clap/summary.json"),
            MetricRef("runs/baselines_musiccaps_full_opus_24kbps/eval_vggish/summary.json"),
            MetricRef("runs/visqol_full_codec/opus_24kbps/summary.json"),
        ),
    }

    same_s_lsd, same_s_mrstft = distance_cols(root, stable_audio3_same_s_clap)
    same_l_lsd, same_l_mrstft = distance_cols(root, stable_audio3_same_l_clap)
    music2latent_lsd, music2latent_mrstft = distance_cols(root, music2latent_clap)
    codicodec_lsd, codicodec_mrstft = distance_cols(root, codicodec_clap)
    codicodec_repro_10k_raw_lsd, codicodec_repro_10k_raw_mrstft = distance_cols(root, codicodec_repro_10k_raw_clap)
    codicodec_repro_10k_ema_lsd, codicodec_repro_10k_ema_mrstft = distance_cols(root, codicodec_repro_10k_ema_clap)
    codicodec_repro_18k_raw_lsd, codicodec_repro_18k_raw_mrstft = distance_cols(root, codicodec_repro_18k_raw_clap)
    codicodec_repro_18k_ema_lsd, codicodec_repro_18k_ema_mrstft = distance_cols(root, codicodec_repro_18k_ema_clap)
    mossland_codec_370k_ema_lsd, mossland_codec_370k_ema_mrstft = distance_cols(root, mossland_codec_370k_ema_clap)
    dac_nq2_lsd, dac_nq2_mrstft = distance_cols(root, dac_rates["1.78 kbps"][0])
    dac_nq3_lsd, dac_nq3_mrstft = distance_cols(root, dac_rates["2.67 kbps"][0])
    dac_nq6_lsd, dac_nq6_mrstft = distance_cols(root, dac_rates["5.33 kbps"][0])
    dac_nq9_lsd, dac_nq9_mrstft = distance_cols(root, dac_clap)

    rows = [
        ReproRow("Musika", "CoDiCodec Table 2 / MusicCaps", "No", "Continuous", "64x", "-", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow("LatMusic", "CoDiCodec Table 2 / MusicCaps", "No", "Continuous", "64x", "-", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow("Moûsai_v2", "CoDiCodec Table 2 / MusicCaps", "Yes", "Continuous", "64x", "-", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow("Moûsai_v3", "CoDiCodec Table 2 / MusicCaps", "Yes", "Continuous", "32x", "-", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow(
            "Music2Latent",
            f"{source}; official SonyCSLParis/music2latent checkpoint",
            "No",
            "Continuous",
            "64x",
            "-",
            "22.05",
            fmt(metric(root, music2latent_clap, "si_sdr_db/mean")),
            fmt(metric(root, music2latent_visqol, "visqol_moslqo/mean")),
            music2latent_lsd,
            music2latent_mrstft,
            fmt(metric(root, music2latent_clap, "fad_clap")),
            fmt(metric(root, music2latent_vggish, "fad_vggish")),
        ),
        ReproRow("Music2Latent2", "CoDiCodec Table 2 / MusicCaps", "Yes", "Continuous", "128x", "-", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow(
            "Stable Audio 3 SAME-S",
            f"{source}; official SAME-S autoencoder",
            "Yes",
            "Continuous",
            "4096-hop / 256-d",
            "-",
            "22.05",
            fmt(metric(root, stable_audio3_same_s_clap, "si_sdr_db/mean")),
            fmt(metric(root, stable_audio3_same_s_visqol, "visqol_moslqo/mean")),
            same_s_lsd,
            same_s_mrstft,
            fmt(metric(root, stable_audio3_same_s_clap, "fad_clap")),
            fmt(metric(root, stable_audio3_same_s_vggish, "fad_vggish")),
        ),
        ReproRow(
            "Stable Audio 3 SAME-L",
            f"{source}; official SAME-L autoencoder",
            "Yes",
            "Continuous",
            "4096-hop / 256-d",
            "-",
            "22.05",
            fmt(metric(root, stable_audio3_same_l_clap, "si_sdr_db/mean")),
            fmt(metric(root, stable_audio3_same_l_visqol, "visqol_moslqo/mean")),
            same_l_lsd,
            same_l_mrstft,
            fmt(metric(root, stable_audio3_same_l_clap, "fad_clap")),
            fmt(metric(root, stable_audio3_same_l_vggish, "fad_vggish")),
        ),
        ReproRow("CoDiCodec (AR)", "CoDiCodec Table 2 / MusicCaps", "Yes", "Continuous", "128x", "-", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow("CoDiCodec (Par., s=3)", "CoDiCodec Table 2 / MusicCaps", "Yes", "Continuous", "128x", "-", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow(
            "CoDiCodec official checkpoint",
            f"{source}; parallel default",
            "Yes",
            "Continuous",
            "128x",
            "-",
            "22.05",
            fmt(metric(root, codicodec_clap, "si_sdr_db/mean")),
            fmt(metric(root, codicodec_visqol, "visqol_moslqo/mean")),
            codicodec_lsd,
            codicodec_mrstft,
            fmt(metric(root, codicodec_clap, "fad_clap")),
            fmt(metric(root, codicodec_vggish, "fad_vggish")),
        ),
        ReproRow(
            "CoDiCodec paper-repro 10k (raw)",
            f"{source}; local 1-10000.ckpt raw; full-clip chunked xfade",
            "Yes",
            "Continuous",
            "128x",
            "-",
            "22.05",
            fmt(metric(root, codicodec_repro_10k_raw_clap, "si_sdr_db/mean")),
            fmt(metric(root, codicodec_repro_10k_raw_visqol, "visqol_moslqo/mean")),
            codicodec_repro_10k_raw_lsd,
            codicodec_repro_10k_raw_mrstft,
            fmt(metric(root, codicodec_repro_10k_raw_clap, "fad_clap")),
            fmt(metric(root, codicodec_repro_10k_raw_vggish, "fad_vggish")),
        ),
        ReproRow(
            "CoDiCodec paper-repro 10k (EMA)",
            f"{source}; local 1-10000.ckpt EMA; full-clip chunked xfade",
            "Yes",
            "Continuous",
            "128x",
            "-",
            "22.05",
            fmt(metric(root, codicodec_repro_10k_ema_clap, "si_sdr_db/mean")),
            fmt(metric(root, codicodec_repro_10k_ema_visqol, "visqol_moslqo/mean")),
            codicodec_repro_10k_ema_lsd,
            codicodec_repro_10k_ema_mrstft,
            fmt(metric(root, codicodec_repro_10k_ema_clap, "fad_clap")),
            fmt(metric(root, codicodec_repro_10k_ema_vggish, "fad_vggish")),
        ),
        ReproRow(
            "CoDiCodec paper-repro 18k (raw)",
            f"{source}; local 2-18000.ckpt raw; full-clip chunked xfade",
            "Yes",
            "Continuous",
            "128x",
            "-",
            "22.05",
            fmt(metric(root, codicodec_repro_18k_raw_clap, "si_sdr_db/mean")),
            fmt(metric(root, codicodec_repro_18k_raw_visqol, "visqol_moslqo/mean")),
            codicodec_repro_18k_raw_lsd,
            codicodec_repro_18k_raw_mrstft,
            fmt(metric(root, codicodec_repro_18k_raw_clap, "fad_clap")),
            fmt(metric(root, codicodec_repro_18k_raw_vggish, "fad_vggish")),
        ),
        ReproRow(
            "CoDiCodec paper-repro 18k (EMA)",
            f"{source}; local 2-18000.ckpt EMA; full-clip chunked xfade",
            "Yes",
            "Continuous",
            "128x",
            "-",
            "22.05",
            fmt(metric(root, codicodec_repro_18k_ema_clap, "si_sdr_db/mean")),
            fmt(metric(root, codicodec_repro_18k_ema_visqol, "visqol_moslqo/mean")),
            codicodec_repro_18k_ema_lsd,
            codicodec_repro_18k_ema_mrstft,
            fmt(metric(root, codicodec_repro_18k_ema_clap, "fad_clap")),
            fmt(metric(root, codicodec_repro_18k_ema_vggish, "fad_vggish")),
        ),
        ReproRow(
            "Mossland codec 370k (EMA)",
            f"{source}; local 2026-06-12_12-46-36 last.ckpt EMA; EncoderDecoder parallel decode",
            "Yes",
            "Continuous",
            "128x",
            "-",
            "22.05",
            fmt(metric(root, mossland_codec_370k_ema_clap, "si_sdr_db/mean")),
            fmt(metric(root, mossland_codec_370k_ema_visqol, "visqol_moslqo/mean")),
            mossland_codec_370k_ema_lsd,
            mossland_codec_370k_ema_mrstft,
            fmt(metric(root, mossland_codec_370k_ema_clap, "fad_clap")),
            fmt(metric(root, mossland_codec_370k_ema_vggish, "fad_vggish")),
        ),
        ReproRow(
            "DAC",
            f"{source}; {dac_rates['2.67 kbps'][3]}",
            "No",
            "Discrete",
            "-",
            "2.67 kbps",
            "22.05",
            fmt(metric(root, dac_rates["2.67 kbps"][0], "si_sdr_db/mean")),
            fmt(metric(root, dac_rates["2.67 kbps"][2], "visqol_moslqo/mean")),
            dac_nq3_lsd,
            dac_nq3_mrstft,
            fmt(metric(root, dac_rates["2.67 kbps"][0], "fad_clap")),
            fmt(metric(root, dac_rates["2.67 kbps"][1], "fad_vggish")),
        ),
        ReproRow(
            "DAC",
            source,
            "No",
            "Discrete",
            "-",
            "8 kbps",
            "22.05",
            fmt(metric(root, dac_clap, "si_sdr_db/mean")),
            fmt(metric(root, dac_visqol, "visqol_moslqo/mean")),
            dac_nq9_lsd,
            dac_nq9_mrstft,
            fmt(metric(root, dac_clap, "fad_clap")),
            fmt(metric(root, dac_vggish, "fad_vggish")),
        ),
        ReproRow("CoDiCodec (AR)", "CoDiCodec Table 2 / MusicCaps", "Yes", "Discrete", "-", "2.38 kbps", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow("CoDiCodec (Par., s=3)", "CoDiCodec Table 2 / MusicCaps", "Yes", "Discrete", "-", "2.38 kbps", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow("CoDiCodec (Par., s=4)", "CoDiCodec Table 2 / MusicCaps", "Yes", "Discrete", "-", "2.38 kbps", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ReproRow(
            "DAC / Proposed",
            f"{source}; {dac_rates['1.78 kbps'][3]}",
            "—",
            "Discrete",
            "—",
            "1.78 kbps",
            "22.05",
            fmt(metric(root, dac_rates["1.78 kbps"][0], "si_sdr_db/mean")),
            fmt(metric(root, dac_rates["1.78 kbps"][2], "visqol_moslqo/mean")),
            dac_nq2_lsd,
            dac_nq2_mrstft,
            fmt(metric(root, dac_rates["1.78 kbps"][0], "fad_clap")),
            fmt(metric(root, dac_rates["1.78 kbps"][1], "fad_vggish")),
        ),
        ReproRow(
            "DAC / Proposed",
            f"{source}; {dac_rates['2.67 kbps'][3]}",
            "—",
            "Discrete",
            "—",
            "2.67 kbps",
            "22.05",
            fmt(metric(root, dac_rates["2.67 kbps"][0], "si_sdr_db/mean")),
            fmt(metric(root, dac_rates["2.67 kbps"][2], "visqol_moslqo/mean")),
            dac_nq3_lsd,
            dac_nq3_mrstft,
            fmt(metric(root, dac_rates["2.67 kbps"][0], "fad_clap")),
            fmt(metric(root, dac_rates["2.67 kbps"][1], "fad_vggish")),
        ),
        ReproRow(
            "DAC / Proposed",
            f"{source}; {dac_rates['5.33 kbps'][3]}",
            "—",
            "Discrete",
            "—",
            "5.33 kbps",
            "22.05",
            fmt(metric(root, dac_rates["5.33 kbps"][0], "si_sdr_db/mean")),
            fmt(metric(root, dac_rates["5.33 kbps"][2], "visqol_moslqo/mean")),
            dac_nq6_lsd,
            dac_nq6_mrstft,
            fmt(metric(root, dac_rates["5.33 kbps"][0], "fad_clap")),
            fmt(metric(root, dac_rates["5.33 kbps"][1], "fad_vggish")),
        ),
        ReproRow(
            "DAC / Proposed",
            f"{source}; {dac_rates['8 kbps'][3]}",
            "—",
            "Discrete",
            "—",
            "8 kbps",
            "22.05",
            fmt(metric(root, dac_clap, "si_sdr_db/mean")),
            fmt(metric(root, dac_visqol, "visqol_moslqo/mean")),
            dac_nq9_lsd,
            dac_nq9_mrstft,
            fmt(metric(root, dac_clap, "fad_clap")),
            fmt(metric(root, dac_vggish, "fad_vggish")),
        ),
        ReproRow("EnCodec", "RVQGAN Table 3 / 44.1 kHz objective eval", "—", "Discrete", "—", "1.5 kbps", "—", not_run, not_run, not_run, not_run, not_run, not_run),
    ]

    for bitrate in ("3 kbps", "6 kbps", "12 kbps", "24 kbps"):
        clap = encodec_claps[bitrate]
        vggish = encodec_vggish[bitrate]
        visqol = fmt(metric(root, encodec_visqol[bitrate], "visqol_moslqo/mean"))
        lsd, mrstft = distance_cols(root, clap)
        rows.append(
            ReproRow(
                "EnCodec",
                source,
                "—",
                "Discrete",
                "—",
                bitrate,
                "22.05",
                fmt(metric(root, clap, "si_sdr_db/mean")),
                visqol,
                lsd,
                mrstft,
                fmt(metric(root, clap, "fad_clap")),
                fmt(metric(root, vggish, "fad_vggish")),
            )
        )

    rows.extend(
        [
            ReproRow("Lyra", "RVQGAN Table 3 / 44.1 kHz objective eval", "—", "Discrete", "—", "9.2 kbps", "—", not_run, not_run, not_run, not_run, not_run, not_run),
        ]
    )
    for bitrate in ("8 kbps", "14 kbps", "24 kbps"):
        clap, vggish, visqol = opus_rates[bitrate]
        lsd, mrstft = distance_cols(root, clap)
        rows.append(
            ReproRow(
                "Opus",
                source,
                "—",
                "Traditional codec",
                "—",
                bitrate,
                "22.05",
                fmt(metric(root, clap, "si_sdr_db/mean")),
                fmt(metric(root, visqol, "visqol_moslqo/mean")),
                lsd,
                mrstft,
                fmt(metric(root, clap, "fad_clap")),
                fmt(metric(root, vggish, "fad_vggish")),
            )
        )
    return rows


def write_csv(rows: list[ReproRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_markdown(rows: list[ReproRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(HEADERS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in HEADERS) + " |\n")
        for row in rows:
            data = row.as_dict()
            handle.write("| " + " | ".join(data[column] for column in HEADERS) + " |\n")


def write_xml(rows: list[ReproRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("<table>\n")
        handle.write("<colgroup>" + "".join('<col width="120"/>' for _ in HEADERS) + "</colgroup>\n")
        handle.write(
            "<thead><tr>"
            + "".join(
                f'<th background-color="light-gray" vertical-align="top"><p>{html.escape(header)}</p></th>'
                for header in HEADERS
            )
            + "</tr></thead>\n"
        )
        handle.write("<tbody>\n")
        for row in rows:
            data = row.as_dict()
            handle.write(
                "<tr>"
                + "".join(
                    f'<td vertical-align="top"><p>{html.escape(data[column])}</p></td>' for column in HEADERS
                )
                + "</tr>\n"
            )
        handle.write("</tbody></table>\n")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows(EVAL_ROOT)
    base = TABLES_DIR / "feishu_reconstruction_repro"
    write_csv(rows, base.with_suffix(".csv"))
    write_markdown(rows, base.with_suffix(".md"))
    write_xml(rows, base.with_suffix(".xml"))
    print(f"wrote {base.with_suffix('.csv')}, {base.with_suffix('.md')}, {base.with_suffix('.xml')}")


if __name__ == "__main__":
    main()
