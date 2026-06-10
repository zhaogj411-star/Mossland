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
        training_wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
        training_wrapper.export_model(ckpt_output_path, export_ema=cfg.export_ema)
    else:
        ckpt_output_path = os.path.join(output_dir, "checkpoint.ckpt")
        experiment_ckpt_path = cfg.experiment_ckpt_path

        checkpoint = torch.load(
            experiment_ckpt_path,
            map_location=training_wrapper.device,
        )
        training_wrapper.load_state_dict(checkpoint["module"], strict=False)
        training_wrapper.export_model(ckpt_output_path, export_ema=cfg.export_ema)


if __name__ == "__main__":
    main()
