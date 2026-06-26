"""Built-in Music-DiT tasks."""

from __future__ import annotations

import torch

from ..core.contract import TaskOutput
from .base import Task
from .registry import register_task


class TextToMusic(Task):
    name = "text2music"
    same_domain = False
    loss_on_gen_only = False

    def apply(self, z0, meta, rng) -> TaskOutput:
        return self._finalize(
            context_latent=self._empty_context(z0),
            cond_mask=self._zeros_mask(z0),
            gen_mask=self._ones_mask(z0),
        )


class Continuation(Task):
    name = "continuation"
    same_domain = True

    def __init__(self, min_ctx: float = 0.25, max_ctx: float = 0.75, side: str = "suffix"):
        self.min_ctx = float(min_ctx)
        self.max_ctx = float(max_ctx)
        self.side = side

    def apply(self, z0, meta, rng) -> TaskOutput:
        frames = z0.shape[-1]
        frac = rng.uniform(self.min_ctx, self.max_ctx)
        keep = max(1, min(frames - 1, int(round(frac * frames))))
        cond_mask = self._zeros_mask(z0)
        if self.side == "suffix":
            cond_mask[:, :keep] = 1.0
        elif self.side == "prefix":
            cond_mask[:, frames - keep :] = 1.0
        else:
            start = (frames - keep) // 2
            cond_mask[:, start : start + keep] = 1.0
        gen_mask = 1.0 - cond_mask
        context = z0 * cond_mask
        return self._finalize(context, cond_mask, gen_mask, info={"ctx_frac": frac, "side": self.side})


class Inpainting(Task):
    name = "inpaint"
    same_domain = True

    def __init__(self, min_hole: float = 0.1, max_hole: float = 0.5, num_holes: int = 1):
        self.min_hole = float(min_hole)
        self.max_hole = float(max_hole)
        self.num_holes = int(num_holes)

    def apply(self, z0, meta, rng) -> TaskOutput:
        frames = z0.shape[-1]
        gen_mask = self._zeros_mask(z0)
        for _ in range(self.num_holes):
            hole = max(1, int(round(rng.uniform(self.min_hole, self.max_hole) * frames / self.num_holes)))
            start = rng.randint(0, max(0, frames - hole))
            gen_mask[:, start : start + hole] = 1.0
        cond_mask = 1.0 - gen_mask
        context = z0 * cond_mask
        return self._finalize(context, cond_mask, gen_mask, info={"num_holes": self.num_holes})


class PairedTranslation(Task):
    name = "paired"
    same_domain = False
    loss_on_gen_only = False

    def __init__(self, context_key: str = "cond_latent", prompt: str | None = None):
        self.context_key = context_key
        self.prompt = prompt

    def apply(self, z0, meta, rng) -> TaskOutput:
        ctx = meta.get(self.context_key)
        if ctx is None:
            raise KeyError(f"{self.name}: meta[{self.context_key!r}] is required")
        ctx = ctx.to(z0.device, z0.dtype)
        if ctx.shape[-1] != z0.shape[-1]:
            ctx = _resize_time(ctx, z0.shape[-1])
        cond_mask = self._ones_mask(z0)
        gen_mask = self._ones_mask(z0)
        return self._finalize(ctx, cond_mask, gen_mask, text_prompt=self.prompt)


class VocalTrackToMusic(PairedTranslation):
    name = "vocaltrack2music"

    def __init__(self, context_key: str = "vocals_latent"):
        super().__init__(context_key=context_key, prompt=None)


class AccToMusic(PairedTranslation):
    name = "acc2music"

    def __init__(self, context_key: str = "accompaniment_latent"):
        super().__init__(context_key=context_key, prompt=None)


class TrackExtraction(PairedTranslation):
    name = "track_extraction"

    def __init__(self, stem: str = "vocals", context_key: str = "mixture_latent"):
        super().__init__(context_key=context_key, prompt=None)
        self.stem = stem

    def apply(self, z0, meta, rng) -> TaskOutput:
        out = super().apply(z0, meta, rng)
        out.info["stem"] = self.stem
        return out


class StyleCover(Task):
    name = "cover"
    same_domain = False
    loss_on_gen_only = False

    def __init__(self, noise_std: float = 1.0, blur: int = 4, context_key: str = "cover_context"):
        self.noise_std = float(noise_std)
        self.blur = int(blur)
        self.context_key = context_key

    def apply(self, z0, meta, rng) -> TaskOutput:
        if meta.get(self.context_key) is not None:
            ctx = meta[self.context_key].to(z0.device, z0.dtype)
            if ctx.shape[-1] != z0.shape[-1]:
                ctx = _resize_time(ctx, z0.shape[-1])
        else:
            ctx = _temporal_blur(z0, self.blur)
            ctx = ctx + self.noise_std * torch.randn_like(ctx)
        cond_mask = self._ones_mask(z0)
        gen_mask = self._ones_mask(z0)
        return self._finalize(ctx, cond_mask, gen_mask, info={"noise_std": self.noise_std})


def _resize_time(z: torch.Tensor, target_t: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        z.unsqueeze(0), size=target_t, mode="linear", align_corners=False
    ).squeeze(0)


def _temporal_blur(z: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return z.clone()
    pad = kernel_size // 2
    padded = torch.nn.functional.pad(z.unsqueeze(0), (pad, pad), mode="reflect")
    weight = z.new_ones(z.shape[0], 1, kernel_size) / kernel_size
    return torch.nn.functional.conv1d(padded, weight, groups=z.shape[0]).squeeze(0)[:, : z.shape[-1]]


register_task("text2music")(lambda **kwargs: TextToMusic(**kwargs))
register_task("vocaltrack2music")(lambda **kwargs: VocalTrackToMusic(**kwargs))
register_task("acc2music")(lambda **kwargs: AccToMusic(**kwargs))
register_task("continuation")(lambda **kwargs: Continuation(**kwargs))
register_task("inpaint")(lambda **kwargs: Inpainting(**kwargs))
register_task("repaint")(lambda **kwargs: Inpainting(**kwargs))
register_task("paired")(lambda **kwargs: PairedTranslation(**kwargs))
register_task("track_extraction")(lambda **kwargs: TrackExtraction(**kwargs))
register_task("cover")(lambda **kwargs: StyleCover(**kwargs))

