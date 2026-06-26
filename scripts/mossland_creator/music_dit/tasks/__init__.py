"""Task library for channel-concat Music-DiT training."""

from .base import Task  # noqa: F401
from .library import (  # noqa: F401
    Continuation,
    Inpainting,
    PairedTranslation,
    StyleCover,
    TextToMusic,
    TrackExtraction,
)
from .registry import available_tasks, build_task, register_task  # noqa: F401
from .sampler import SampledTask, TaskMixSampler, TaskSpec  # noqa: F401

