import os
import yaml
from hydra.utils import instantiate
import torch
import time
import hydra
from omegaconf import OmegaConf

# from scripts.audio_autoencoder.utils import remove_weight_norm_from_model


def remove_weightnorm(model):
    """
    移除模型中的weightnorm
    """
    for module in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
        except ValueError:  # 如果模块没有weightnorm，跳过
            pass
    return model


def load_model(ckpt_path=None, config_path=None, ckpt_dir=None):
    # 如果提供了 ckpt_dir，则从中加载 config.yaml 和检查点文件
    if ckpt_dir is not None:
        config_path = os.path.join(ckpt_dir, "config.yaml")
        ckpt_path = os.path.join(
            ckpt_dir, "checkpoint.ckpt")  # 假设检查点文件名为 checkpoint.ckpt

    # 加载检查点信息
    ckpt_info = torch.load(ckpt_path,weights_only=False)

    # 加载配置文件
    if config_path is None:
        config = ckpt_info["config"]
    else:
        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

    # 实例化模型
    model = instantiate(config)
    # model = remove_weight_norm_from_model(model)
    
    # 加载模型状态字典
    if "state_dict" in ckpt_info and isinstance(ckpt_info["state_dict"], dict):
        state_dict = ckpt_info["state_dict"]
    else:
        # 如果检查点本身就是状态字典
        state_dict = ckpt_info
    
    # print(state_dict)
    model.load_state_dict(state_dict, strict=True)

    # 返回模型
    return model

if __name__ == "__main__":
    model = load_model(
        "/home/gjzhao/workspace2024/heartstrings-version4/ckpt/loop_prompting_diffusion_48000_new.ckpt"
    )

    print(1)
