# Copyright (c) 2025.
"""Pluggable audio codecs. Import side-effect: registers built-in codecs."""

from .base import AudioCodec, LatentStats  # noqa: F401
from .registry import available_codecs, build_codec, register_codec  # noqa: F401

from . import dummy  # noqa: F401,E402
from . import oobleck  # noqa: F401,E402
from . import codicodec  # noqa: F401,E402
from . import same  # noqa: F401,E402
