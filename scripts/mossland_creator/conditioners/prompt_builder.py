# Copyright (c) 2025.
"""Turn a Sonata parquet row (metadata dict) into a natural-language prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _coerce(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s and s[0] in "[{":
            try:
                return json.loads(s)
            except Exception:
                return value
    return value


@dataclass
class PromptBuilderConfig:
    use_caption: bool = True
    use_key: bool = True
    use_bpm: bool = True
    use_instruments: bool = True
    use_structure: bool = False
    use_vocal_flag: bool = True
    caption_field: str = "caption"
    key_field: str = "key"
    bpm_field: str = "bpm"
    instruments_field: str = "instruments"
    structure_field: str = "Song structure"
    has_vocals_field: str = "has_vocals_Qwen3-Omni"
    max_instruments: int = 6
    field_dropout: float = 0.0
    separator: str = ". "


class PromptBuilder:
    def __init__(self, config: Optional[PromptBuilderConfig] = None):
        self.cfg = config or PromptBuilderConfig()

    def build(self, row: Dict[str, Any], rng=None) -> str:
        cfg = self.cfg
        parts: List[str] = []

        def keep(p: float) -> bool:
            if cfg.field_dropout <= 0 or rng is None:
                return True
            return rng.random() >= p

        if cfg.use_caption and row.get(cfg.caption_field) and keep(cfg.field_dropout):
            parts.append(str(row[cfg.caption_field]).strip())

        if cfg.use_key and keep(cfg.field_dropout):
            key = _coerce(row.get(cfg.key_field))
            if isinstance(key, dict) and key.get("key"):
                scale = key.get("scale", "")
                parts.append(f"key {key['key']} {scale}".strip())

        if cfg.use_bpm and row.get(cfg.bpm_field) and keep(cfg.field_dropout):
            try:
                parts.append(f"{round(float(row[cfg.bpm_field]))} bpm")
            except Exception:
                pass

        if cfg.use_instruments and keep(cfg.field_dropout):
            inst = _coerce(row.get(cfg.instruments_field))
            names = self._instrument_names(inst)
            if names:
                parts.append("instruments: " + ", ".join(names[: cfg.max_instruments]))

        if cfg.use_vocal_flag and keep(cfg.field_dropout):
            hv = row.get(cfg.has_vocals_field)
            if hv in ("是", True, "true", "True", "yes"):
                parts.append("with vocals")
            elif hv in ("否", False, "false", "False", "no"):
                parts.append("instrumental")

        if cfg.use_structure and keep(cfg.field_dropout):
            struct = _coerce(row.get(cfg.structure_field))
            labels = self._structure_labels(struct)
            if labels:
                parts.append("structure: " + " ".join(labels))

        return cfg.separator.join(p for p in parts if p)

    @staticmethod
    def _instrument_names(inst) -> List[str]:
        names: List[str] = []
        if isinstance(inst, list):
            seen = set()
            for frame in inst:
                active = frame.get("active", []) if isinstance(frame, dict) else []
                for a in active:
                    if a not in seen:
                        seen.add(a)
                        names.append(a)
        return names

    @staticmethod
    def _structure_labels(struct) -> List[str]:
        out: List[str] = []
        if isinstance(struct, list):
            prev = None
            for seg in struct:
                lab = seg.get("label") if isinstance(seg, dict) else None
                if lab and lab != prev:
                    out.append(f"[{lab}]")
                    prev = lab
        return out
