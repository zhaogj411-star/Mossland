from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as pl
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from scripts.trainer_utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
import os

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    ##cfg的额外设置
    extras(cfg)
    OmegaConf.resolve(cfg)  # resolve all string interpolations
    ##logger
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    # dataset
    log.info(f"Instantiating data module <{cfg.data._target_}>")
    data = hydra.utils.instantiate(cfg.data)
    data.setup()
    # model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    if "pretrain_ckpt_path" in cfg and cfg["pretrain_ckpt_path"] is not None:
        model_state_dict = torch.load(
            cfg["pretrain_ckpt_path"], map_location="cpu", weights_only=False
        )["state_dict"]
        # breakpoint()
        # 尝试加载模型权重，跳过不匹配的参数
        missing_keys, unexpected_keys = [], []
        for name, param in model.named_parameters():
            if name in model_state_dict and param.shape == model_state_dict[name].shape:
                param.data.copy_(model_state_dict[name])
            else:
                missing_keys.append(name)
        log.info(f"加载预训练模型权重，跳过了 {len(missing_keys)} 个不匹配的参数")
        # model.load_state_dict(model_state_dict, strict=True)
    # training wrappr
    log.info(f"Instantiating model <{cfg.wrapper._target_}>")

    training_wrapper = hydra.utils.instantiate(cfg.wrapper, model=model)

    # checkpoint = torch.load(
    #     "/home/gjzhao/workspace2024/heartstrings-version1/logs/vae_large/runs/2024-07-22_15-26-55/checkpoints/last.ckpt",
    #     map_location=training_wrapper.device,
    # )
    # training_wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    ## callbacks
    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.callbacks)
    ##trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    object_dict = {
        "cfg": cfg,
        "datamodule": data,
        "wrapper": training_wrapper,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    log.info("Starting training!")
    if cfg.resume_from_ckpt is not None:
        if os.path.exists(cfg.resume_from_ckpt):
            print("checkpoint存在")
            trainer.fit(
                training_wrapper, datamodule=data, ckpt_path=cfg.resume_from_ckpt
            )
        else:
            print("checkpoint不存在")

            trainer.fit(training_wrapper, datamodule=data)
    else:
        trainer.fit(training_wrapper, datamodule=data)


if __name__ == "__main__":
    main()
