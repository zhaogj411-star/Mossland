"""codec 实验共享的训练/量化模块单一来源。

历史上 quantize.py / training_base.py 在 mossland_codec、music2latent、
same_flow、same_flow_debug 下逐字节复制多份。现统一收于此处，各实验目录从这里导入。
"""

from .quantize import ResidualVectorQuantize
from .training_base import (
    CodecTrainingBase,
    add_noise,
    get_sigma_continuous,
    pseudo_huber_loss,
)

__all__ = [
    "ResidualVectorQuantize",
    "CodecTrainingBase",
    "add_noise",
    "get_sigma_continuous",
    "pseudo_huber_loss",
]
