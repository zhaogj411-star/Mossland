from typing import Any, Dict, List, Optional, Tuple
import os

import hydra
import lightning as pl
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import shutil
from scripts.trainer_utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _partial_codebook_copy(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor | None:
    if target.ndim != source.ndim or target.ndim < 2:
        return None
    if target.shape[0] != source.shape[0]:
        return None
    if target.shape[2:] != source.shape[2:]:
        return None
    copied = target.detach().clone()
    n = min(target.shape[1], source.shape[1])
    copied[:, :n, ...] = source[:, :n, ...].to(device=copied.device, dtype=copied.dtype)
    return copied


def _load_state_dict_flexible(module: torch.nn.Module, state_dict: dict, strict: bool):
    if strict:
        return module.load_state_dict(state_dict, strict=True)

    current = module.state_dict()
    filtered = {}
    copied_partial = []
    skipped_shape = []

    for key, value in state_dict.items():
        if key not in current:
            filtered[key] = value
            continue
        target = current[key]
        if tuple(value.shape) == tuple(target.shape):
            filtered[key] = value
            continue
        partial = _partial_codebook_copy(target, value)
        if partial is not None:
            filtered[key] = partial
            copied_partial.append(
                f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}"
            )
        else:
            skipped_shape.append(
                f"{key}: {tuple(value.shape)} -> {tuple(target.shape)}"
            )

    incompatible = module.load_state_dict(filtered, strict=False)
    if copied_partial:
        log.info(
            "Partially copied mismatched codebook tensors; old entries copied, "
            f"new entries kept from fresh init: {len(copied_partial)}"
        )
        for item in copied_partial[:20]:
            log.info(f"  partial_copy {item}")
        if len(copied_partial) > 20:
            log.info(f"  ... {len(copied_partial) - 20} more partial copies")
    if skipped_shape:
        log.info(f"Skipped shape-mismatched tensors: {len(skipped_shape)}")
        for item in skipped_shape[:20]:
            log.info(f"  skip_shape {item}")
        if len(skipped_shape) > 20:
            log.info(f"  ... {len(skipped_shape) - 20} more skipped shape mismatches")
    return incompatible


@hydra.main(version_base=None, config_path="./configs", config_name="unwrap_model")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # resolve all string interpolations
    # model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    # training wrapper
    log.info(f"Instantiating model <{cfg.wrapper._target_}>")
    training_wrapper = hydra.utils.instantiate(cfg.wrapper, model=model)

    # 创建文件夹保存配置和权重文件
    output_dir = cfg.output_path
    os.makedirs(output_dir, exist_ok=True)
    load_strict = bool(cfg.get("load_strict", True))

    # 保存配置文件
    config_file_path = os.path.join(output_dir, "config.yaml")
    with open(config_file_path, "w") as config_file:
        OmegaConf.save(cfg.model, config_file)
    deepspeed = cfg.deepspeed
    # 路径
    if not deepspeed:
        experiment_ckpt_path = cfg.experiment_ckpt_path
        ckpt_output_path = os.path.join(output_dir, "checkpoint.ckpt")

        checkpoint = torch.load(
            experiment_ckpt_path,
            map_location=training_wrapper.device,
        )
        incompatible = _load_state_dict_flexible(
            training_wrapper,
            checkpoint["state_dict"],
            strict=load_strict,
        )
        if not load_strict:
            log.info(
                "Loaded checkpoint with strict=False: "
                f"missing_keys={len(incompatible.missing_keys)}, "
                f"unexpected_keys={len(incompatible.unexpected_keys)}"
            )
        training_wrapper.export_model(ckpt_output_path, export_ema=cfg.export_ema)
    else:
        ckpt_output_path = os.path.join(output_dir, "checkpoint.ckpt")
        experiment_ckpt_path = cfg.experiment_ckpt_path

        checkpoint = torch.load(
            experiment_ckpt_path,
            map_location=training_wrapper.device,
        )
        incompatible = _load_state_dict_flexible(
            training_wrapper,
            checkpoint["module"],
            strict=load_strict,
        )
        if not load_strict:
            log.info(
                "Loaded checkpoint with strict=False: "
                f"missing_keys={len(incompatible.missing_keys)}, "
                f"unexpected_keys={len(incompatible.unexpected_keys)}"
            )
        training_wrapper.export_model(ckpt_output_path, export_ema=cfg.export_ema)


if __name__ == "__main__":
    main()
