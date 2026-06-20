from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as AF

from .metrics import frechet_distance, mel_frechet_embedding, mono_fold_down


@dataclass(frozen=True)
class FadResult:
    metric_name: str
    backend_name: str
    reference_embedding: torch.Tensor
    prediction_embedding: torch.Tensor


class FadBackend:
    metric_name = "fad"
    backend_name = "base"

    def embed(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        raise NotImplementedError

    def pair(self, reference: torch.Tensor, prediction: torch.Tensor, sample_rate: int) -> FadResult:
        return FadResult(
            metric_name=self.metric_name,
            backend_name=self.backend_name,
            reference_embedding=self.embed(reference, sample_rate).detach().cpu(),
            prediction_embedding=self.embed(prediction, sample_rate).detach().cpu(),
        )


class NoFadBackend(FadBackend):
    metric_name = "fad_disabled"
    backend_name = "none"

    def embed(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return torch.empty(0)

    def pair(self, reference: torch.Tensor, prediction: torch.Tensor, sample_rate: int) -> FadResult:
        return FadResult(
            metric_name=self.metric_name,
            backend_name=self.backend_name,
            reference_embedding=torch.empty(0),
            prediction_embedding=torch.empty(0),
        )


class MelProxyFadBackend(FadBackend):
    metric_name = "fad_mel_proxy"
    backend_name = "mel_log_stats_proxy"

    def __init__(self, device: str | torch.device | None = None):
        self.device = device

    def embed(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.device is not None:
            audio = audio.to(self.device, non_blocking=True)
        return mel_frechet_embedding(audio, sample_rate)


class ClapFadBackend(FadBackend):
    """CLAP-feature Frechet distance compatible with the CoDiCodec FAD_clap metric.

    CoDiCodec reports FAD_clap as a Frechet distance over CLAP features. This
    backend uses Hugging Face's LAION CLAP audio encoder. It is intentionally
    optional because the model is large and requires downloading weights.
    """

    metric_name = "fad_clap"
    backend_name = "laion_clap_hf"
    sample_rate = 48000

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        cache_dir: str | Path = "checkpoints/clap",
        device: str | torch.device | None = None,
    ):
        from transformers import ClapModel, ClapProcessor

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.processor = ClapProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
            self.model = ClapModel.from_pretrained(model_name, cache_dir=str(cache_dir)).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load CLAP FAD backend. Download the Hugging Face "
                f"model {model_name!r} into checkpoints/clap or fix HF_ENDPOINT/"
                "proxy access, then rerun with --fad-backend clap. "
                "Use --fad-backend mel_proxy for a lightweight non-paper proxy."
            ) from exc

    def _prepare_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        audio = mono_fold_down(audio.float()).squeeze(0)
        if sample_rate != self.sample_rate:
            audio = AF.resample(audio.unsqueeze(0), sample_rate, self.sample_rate).squeeze(0)
        return audio.clamp(-1.0, 1.0)

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        mono = self._prepare_audio(audio.detach().cpu(), sample_rate)
        inputs = self.processor(
            audios=mono.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        features = self.model.get_audio_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1).squeeze(0).detach().cpu()


class VggishFadBackend(FadBackend):
    """VGGish-feature Frechet Audio Distance backend.

    Google Research's original FAD implementation extracts 128-D raw VGGish
    embeddings before the PCA/quantization postprocessor. This backend uses the
    PyTorch port in `harritaylor/torchvggish`, whose weights are ported from the
    TensorFlow VGGish checkpoint. Each audio item can yield multiple 0.96 s
    embeddings; the per-item embedding is their mean so the existing row-level
    FAD summarizer can keep one vector per manifest row.
    """

    metric_name = "fad_vggish"
    backend_name = "torchvggish_raw"
    sample_rate = 16000
    min_samples = 16000

    def __init__(
        self,
        model_name: str | Path = "tmp/eval_metric_refs/torchvggish",
        cache_dir: str | Path = "checkpoints/vggish",
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = str(model_name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        source = "local" if Path(self.model_name).exists() else "github"
        previous_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(str(self.cache_dir / "torch_hub"))
        try:
            self.model = torch.hub.load(
                self.model_name,
                "vggish",
                source=source,
                postprocess=False,
                progress=False,
            )
        except Exception as exc:  # pragma: no cover - network/checkpoint dependent
            raise RuntimeError(
                "Failed to load VGGish FAD backend. Clone harritaylor/torchvggish "
                "to tmp/eval_metric_refs/torchvggish or pass --fad-model with a "
                "local torchvggish repo path. If weights are missing, retry with "
                "proxy variables cleared so torch.hub can download vggish-10086976.pth."
            ) from exc
        finally:
            torch.hub.set_dir(previous_hub_dir)

        # The TensorFlow AudioSet/FAD wrapper uses the pre-activation tensor
        # `vggish/embedding:0`. The PyTorch port includes a final ReLU in the
        # embeddings Sequential, so remove it when present.
        embeddings = getattr(self.model, "embeddings", None)
        if isinstance(embeddings, torch.nn.Sequential) and len(embeddings) > 0:
            children = list(embeddings.children())
            if isinstance(children[-1], torch.nn.ReLU):
                self.model.embeddings = torch.nn.Sequential(*children[:-1])
        self.model.eval().to(self.device)
        if hasattr(self.model, "device"):
            self.model.device = self.device

    def _prepare_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        audio = mono_fold_down(audio.float()).squeeze(0)
        if sample_rate != self.sample_rate:
            audio = AF.resample(audio.unsqueeze(0), sample_rate, self.sample_rate).squeeze(0)
        audio = audio.clamp(-1.0, 1.0)
        if audio.numel() < self.min_samples:
            audio = torch.nn.functional.pad(audio, (0, self.min_samples - audio.numel()))
        return audio

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        mono = self._prepare_audio(audio.detach().cpu(), sample_rate)
        # torchvggish preprocessing accepts numpy waveform input and returns
        # one embedding per VGGish example window.
        embeddings = self.model.forward(mono.numpy(), self.sample_rate)
        embeddings = torch.as_tensor(embeddings).detach().float().cpu()
        if embeddings.ndim == 1:
            return embeddings
        if embeddings.ndim != 2:
            raise RuntimeError(f"VGGish returned unexpected embedding shape {tuple(embeddings.shape)}.")
        return embeddings.mean(dim=0)


def build_fad_backend(
    backend: str,
    device: str | torch.device | None = None,
    model_name: str | None = None,
    cache_dir: str | Path | None = None,
) -> FadBackend:
    if backend == "none":
        return NoFadBackend()
    if backend == "mel_proxy":
        return MelProxyFadBackend(device=device)
    if backend == "clap":
        return ClapFadBackend(
            model_name=model_name or "laion/clap-htsat-unfused",
            cache_dir=cache_dir or "checkpoints/clap",
            device=device,
        )
    if backend == "vggish":
        return VggishFadBackend(
            model_name=model_name or "tmp/eval_metric_refs/torchvggish",
            cache_dir=cache_dir or "checkpoints/vggish",
            device=device,
        )
    raise ValueError(f"unsupported FAD backend: {backend}")


def summarize_fad(rows: list[dict], metric_name: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[torch.Tensor]]] = {}
    for row in rows:
        ref_embedding = row.get("reference_fad_embedding")
        pred_embedding = row.get("prediction_fad_embedding")
        if ref_embedding is None or pred_embedding is None:
            continue
        if len(ref_embedding) == 0 or len(pred_embedding) == 0:
            continue
        key = row["task_id"]
        if row.get("low_sample_rate"):
            key = f"{key}@{row['low_sample_rate']}"
        group = grouped.setdefault(key, {"reference": [], "prediction": []})
        group["reference"].append(torch.tensor(ref_embedding, dtype=torch.float64))
        group["prediction"].append(torch.tensor(pred_embedding, dtype=torch.float64))

    summary = {}
    for key, features in grouped.items():
        if len(features["reference"]) < 2:
            summary[key] = {metric_name: float("nan"), "count": float(len(features["reference"]))}
            continue
        summary[key] = {
            metric_name: frechet_distance(
                torch.stack(features["reference"], dim=0),
                torch.stack(features["prediction"], dim=0),
            ),
            "count": float(len(features["reference"])),
        }
    return summary


def row_fad_backend(rows: list[dict]) -> str | None:
    for row in rows:
        value: Any = row.get("fad_embedding_backend")
        if value:
            return str(value)
    return None
