# SAME 训练实现记录

## 当前实现

- `scripts/same/wrapper.py` 已从简单 waveform/STFT loss 改为公开 Stable Audio Tools autoencoder wrapper 的核心训练结构：Lightning manual optimization，生成器步和判别器步交替。
- `scripts/configs/experiment/same.yaml` 的 `model:` 只表达模型结构；SAME-L 初始化权重通过顶层 `pretrain_ckpt_path` 加载，不在 `model:` 下放 `ckpt_dir` 或 `scripts.factory.load_model`。
- `wrapper.loss_config` 必须完全来自 YAML；代码里不再保留 `default_loss_config()` 或旧的顶层 `waveform_loss_weight/stft_loss_weight/softnorm_loss_weight` 覆盖入口，避免训练口径隐藏在 Python 默认值里。
- 因为使用 manual optimization，SAME 实验配置必须关闭 Lightning Trainer 自动梯度裁剪；`scripts/configs/experiment/same.yaml` 覆盖 `trainer.gradient_clip_val=null`、`trainer.gradient_clip_algorithm=null`，实际裁剪由 wrapper 参数 `clip_grad_norm` 手动执行。
- 频谱重建 loss 使用官方公开代码默认的 7 尺度 MRSTFT：`[2048,1024,512,256,128,64,32]`，75% overlap，`perceptual_weighting=true`、`w_sc=1.0`、`w_log_mag=1.0`、`w_lin_mag=0.0`。官方 auraloss fork 支持 `w_phs` phase loss，但默认是 `0.0`，当前 `same.yaml` 不启用 phase 项。
- stereo 训练按官方 wrapper 同时计算 mid/side (`SumAndDifferenceSTFTLoss`) 和 left/right (`MultiResolutionSTFTLoss`)；两者共用同一个 YAML 权重 `spectral.weights.mrstft`。
- adversarial 训练使用自包含 EnCodec 风格 multi-scale STFT discriminator，支持 relativistic paired GAN (`rpgan`) 和 feature matching；当前 YAML 权重为 adversarial `0.1`、feature matching `5.0`。
- `scripts/configs/experiment/same.yaml` 已显式写出上述 loss/discriminator 配置，SoftNorm 权重按官方 fallback 写为 `1e-5`，`checkpoint_monitor` 改为 `train/loss`。

## 尚未完全覆盖论文的部分

- 论文阶段式训练（Stage 1 全 AE、Stage 2 decoder finetune、Stage 3 transformer discriminator + chirps）还没有写成独立 Hydra 配置组；当前 wrapper 支持 `decoder_finetune`，但没有自动重置判别器或自动追加 chirps。
- 论文提到的 multi-view discriminator 里的 PQMF/chroma 视图尚未移植；当前是官方公开 wrapper 默认的 EnCodec MS-STFT discriminator 核心。
- latent diffusion auxiliary、semantic chroma/ILD regressors、contrastive/JEPA auxiliary 没有在 `scripts/same` 自包含实现中打开；官方 public wrapper 有部分参考代码，但依赖更多 Stable Audio Tools 模块，后续若要完全追论文需继续移植。
- transformer discriminator stage 尚未移植。

## 验证

- `python -m py_compile scripts/same/losses.py scripts/same/discriminators.py scripts/same/wrapper.py`
- 随机 stereo tensor 上 `SumAndDifferenceSTFTLoss` 与 `EncodecDiscriminator.loss()` smoke 通过。
- 极小 SAME autoencoder 在 CPU 上用 Lightning 跑 `max_steps=2` 通过，覆盖生成器步和判别器步。
- Hydra compose `experiment=same trainer.devices=8` 确认最终 `trainer.gradient_clip_val=None`、`trainer.gradient_clip_algorithm=None`、`wrapper.clip_grad_norm=1.0`。
- Hydra instantiate SAME-L 结构后与 `pretrain_ckpt_path` 的 `ckpt/SAME-L/checkpoint.ckpt` 对比：`model_keys=472`、`ckpt_keys=472`、`matched=472`、`missing=0`、`skipped=0`、参数量 `852127073`。
