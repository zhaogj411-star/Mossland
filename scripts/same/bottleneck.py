import torch
from torch import nn


class SoftNormBottleneck(nn.Module):
    def __init__(
        self,
        dim=32,
        noise_augment_dim=0,
        noise_regularize=False,
        auto_scale=False,
        freeze=False,
    ):
        super().__init__()
        self.noise_augment_dim = int(noise_augment_dim)
        self.scaling_factor = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))
        self.noise_scaling_factor = nn.Parameter(torch.ones(1, self.noise_augment_dim, 1))
        self.noise_regularize = bool(noise_regularize)
        self.freeze = bool(freeze)
        if self.freeze:
            self.scaling_factor.requires_grad = False
            self.bias.requires_grad = False
            self.noise_scaling_factor.requires_grad = False
        if auto_scale:
            self.register_parameter(
                "running_std", nn.Parameter(torch.ones(1), requires_grad=False)
            )

    def encode(self, x, return_info=False, **kwargs):
        info = {}
        x = x * self.scaling_factor + self.bias
        if self.training and hasattr(self, "running_std") and not self.freeze:
            self.running_std.data = (
                self.running_std.data * 0.999 + x.std().detach() * 0.001
            ).clamp(min=1e-4)
        if hasattr(self, "running_std"):
            x = x / self.running_std
        if self.training and return_info:
            var_t = (x.std(dim=-1) ** 2).clip(min=1e-4)
            mean_t = x.mean(dim=-1)
            loss = (mean_t * mean_t + var_t - torch.log(var_t) - 1).mean()
            var_c = (x.std(dim=-2) ** 2).clip(min=1e-4)
            mean_c = x.mean(dim=-2)
            loss = loss + 0.4 * (mean_c * mean_c + var_c - torch.log(var_c) - 1).mean()
            info["softnorm_loss"] = loss
        if return_info:
            return x, info
        return x

    def decode(self, x, **kwargs):
        if hasattr(self, "running_std"):
            x = x * self.running_std
        if self.noise_regularize:
            scaling = self.running_std if hasattr(self, "running_std") else x.std(dim=-1).unsqueeze(-1)
            scale = 5e-2 if self.training else 1e-3
            x = x + torch.randn_like(x) * scaling * scale
        if self.noise_augment_dim > 0:
            noise = self.noise_scaling_factor * torch.randn(
                x.shape[0], self.noise_augment_dim, x.shape[-1], device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, noise], dim=1)
        return x

