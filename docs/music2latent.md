# Music2Latent 训练包说明

本文件记录 `scripts/music2latent` 的当前边界、训练入口和配置约定。

## 迁移边界

- `scripts/music2latent/` 是从 `tmp/music2latent-training/music2latent-training` 官方训练代码整理来的独立 Hydra/Lightning 训练包。
- 不复制官方 `launch.py`、`config_loader.py`、`hparams.py`、`hparams_inference.py`、notebook、图片、`setup.py` 或 FAD 脚本；参数通过 `scripts/configs/experiment/music2latent.yaml` 的 `model:` / `wrapper:` 直接传入。
- `scripts/mossland-codec/` 不 import `scripts/music2latent`；两者是并列训练实现。

## 代码结构

- `audio.py`：STFT real/imag 表示和 inverse STFT。支持双声道输入 `[B, 2, T]`，表示通道为 `2 * audio_channels`。
- `diffusion.py`：EDM/consistency 训练所需的 sigma、noise、step schedule 和数值稳定 pseudo-Huber loss。
- `models.py`：`Music2LatentUNet`，构造函数直接接收 YAML 中的参数；没有内部 config dataclass，也不读取全局 `hparams`。模型输出 continuous latent，并可选通过 `vector_quantize_pytorch.ResidualVQ` 得到 discrete/RVQ latent。
- `local_transformer.py`：从 `scripts/same/transformer.py` 复制/适配的 sliding-window SDPA、DyT/RMSNorm、RoPE 和 TransformerBlock，并封装 `AxialLocalAttention2d` / `AxialLocalAttentionResBlock`，输入输出保持 Music2Latent 的 `[B,C,F,T]` channel-first 约定。
- `wrapper.py`：`Music2LatentTrainingWrapper` 接入 Lightning 训练，负责 sigma sampling、continuous/RVQ path、RAdam、cosine+warmup、EMA、loss/latent logging 和 finite diagnostics；`Music2LatentTrainingCallback` 负责 demo。

## 训练配置

训练入口：

```sh
PYTHONPATH=$PWD python -m scripts.train experiment=music2latent
```

`scripts/configs/experiment/music2latent.yaml` 当前使用 `scripts.music2latent.data.Music2LatentOfficialPTDataset` 读取预裁好的 `.pt` mono waveform。2026-06-20 本机训练口径指向 `tmp/music2latent_official_pt_10k/files.list`；该目录已有 `000000.pt` 到 `009999.pt` 共 10000 个预处理样本，当前清单启用前 1000 条。每条 `audio` 是官方短窗长度 `34304` samples 的 44.1 kHz mono `float32` crop，源自本机 Sonata `NETEASE_SPIDER/audio/.../*.mp3`。`data.dataset.length=160000` 保持原训练逻辑，会循环这 1000 条物理样本。

模型参数按官方 Music2Latent 训练口径设置：`hop=512`、`fac=4`、`data_length=64`；官方短窗长度是 `hop * data_length + (fac - 1) * hop = 34304`。当前 2 秒本机样本更长，wrapper 的 `training_step()` 会把 waveform 裁到合法长度，使 STFT frame 数可被 `2 ** freq_downsample_list.count(0)` 整除；例如默认 `[1,0,0,0]` 要求 frame 数是 8 的倍数，避免长音频出现 decoder skip feature 的 `582` vs `584` 这类 shape mismatch。

`scripts/configs/experiment/music2latent_axiallocal.yaml` 是 SAME-style axial local denoiser 试验入口。它保持官方 `Music2Latent` 的 encoder、latent pyramid decoder、down/up resampling、wrapper 和 loss，把 denoising U-Net 中 `out_channels >= axial_min_channels` 的 2D `ResBlock` 替换为 `Music2Latent_axiallocal` 中的 `AxialLocalAttentionResBlock`；当前默认 `axial_min_channels=256`，只覆盖低分辨率/高通道阶段，避免在最高分辨率 `[B,64,1024,T]` 特征图上直接构造整轴 Q/K/V。默认窗口为 `axial_time_window=[1,1]`、`axial_freq_window=[1,1]`，不依赖 NATTEN；如果需要诊断全量替换，显式覆盖 `model.axial_min_channels=0`。

`scripts/configs/experiment/music2latent_localattention.yaml` 当前不是 NATTEN 全卷积替换版，而是从 `music2latent.yaml` 出发的 SAME temporal-only 入口。模型 target 为 `scripts.music2latent.models.Music2Latent_sametemporal`：保留官方 2D encoder/denoising U-Net/latent pyramid/frequency resampling，只把 encoder/decoder bottleneck 中有时间局部计算的 `ResBlock(use_2d=False)` 替换为直接来自 `scripts/same/transformer.py` 的 sliding-window `TransformerBlock`。默认 `same_sliding_window=[1,1]`，不允许 full attention；`Conv1d(kernel=1)` channel projection、`Conv2d(kernel=3)` 和频率 `(5,1)` 卷积保持原版线性/卷积实现。

`scripts/configs/experiment/music2latent_rvq.yaml` 是基于 `music2latent.yaml` 的 EMA RVQ 版本。它保持官方模型、数据、trainer 和 consistency wrapper 口径，打开 `quantizer_num_quantizers=32`、`quantizer_codebook_size=1024`、`quantizer_codebook_dim=null`、`quantizer_dropout=1.0`，用于训练期支持动态码率量化。训练和 demo 均使用 `[8,16,32]` 个 quantizer 选项；demo callback 会把 `source -> discrete8 -> discrete16 -> discrete32 -> continuous` 拼成同一个 mp3 保存。

`scripts/configs/experiment/music2latent_rvq_0p5b_stereo.yaml` 是基于 RVQ 配置的新 0.5B 双声道版本。模型使用 `data_channels=4`、`base_channels=192`、`bottleneck_base_channels=1536`、`bottleneck_channels=192`、`cond_channels=768`、`heads=8`，实际实例化参数量约 `515.13M`。RVQ 使用 32 个 1024-size codebook，`quantizer_codebook_dim=16`、`quantizer_dropout=1.0`，训练和 demo 码率选项为 `[8,16,32]`。正式训练数据直接用 `scripts.data.datasets.SampleDataset` 从 `/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER` 读取 raw audio，随机裁 8 秒 stereo：`sample_size=352800`、`num_channels=2`、`length=10000000`、`train_batch_size=1`。

## RVQ 训练路径

当 `quantizer_num_quantizers > 0` 时，`Music2Latent.quantize_representation()` 会先通过 encoder 得到 `continuous` latent 和 encoder hidden，再把 hidden 送入 EMA RVQ，得到 `discrete` latent。`wrapper.rvq_latent_train_prob` 现在是 batch 内逐样本混合概率：每个样本独立采样一个 mask，mask 为 true 时把 discrete latent 作为 denoiser conditioning，否则使用 continuous latent；`latent/source_discrete` 记录本 batch 中 discrete 样本比例。`wrapper.rvq_detach_encoder=false` 时，RVQ 输入不再 detach，选中 discrete 的样本会让 consistency loss 通过 RVQ straight-through 路径回到 encoder hidden；正式 RVQ 配置当前使用 `rvq_latent_train_prob=0.25`、`rvq_detach_encoder=false`，让离散路径参与主干调整但不完全替代 continuous 训练。

训练时 `train_n_quantizers_choices=[8,16,32]` 会按 step 采样本次 RVQ 深度，continuous/discrete batch 混合共享同一个采样深度。`rvq_commitment_weight`、`rvq_codebook_weight` 和 `rvq_hidden_recon_weight` 当前仍为 0，codebook 主要靠 EMA assignment/update 维护；主网络适配 discrete path 的梯度来自被 mask 选中的 discrete consistency loss，而不是额外的 RVQ aux loss。

## 数值诊断

`pseudo_huber_loss()` 使用 `torch.hypot(predicted - target, delta)`，避免大残差先平方导致 float32 overflow。wrapper 在训练中检查 representation、continuous/RVQ latents、RVQ loss、sigma、consistency weight/loss 和最终 loss 的 finite 状态；`on_after_backward()` 会在梯度出现 NaN/Inf 时报告具体参数名。

2026-06-18 64 卡 run `logs/music2latent/runs/2026-06-17_18-00-53` 在 `global_step=438` 崩溃，首因是多个 rank 报 `FloatingPointError: Non-finite gradient encoder.gain.scale`；NCCL/TCPStore/`Closing env plugin` 信息是 rank 退出后的清理噪声。TensorBoard 在 step 429 仍显示 `loss=0.0220`、`loss/consistency=0.0218`、`latent/std=0.450`、`sigma/step≈0.09975`，说明 forward loss 和已记录标量未先变 NaN。`encoder.gain.scale` 来自 `FreqGain`，位于 encoder `conv_inp` 后、下采样前；当前 64 卡配置为每卡 batch 2、`sample_size=1000000`、bf16 mixed，长时间频率图上的梯度聚合面很大，后续应优先验证禁用 `model.frequency_scaling`、降低 warmup 峰值学习率或给 `FreqGain` 增加单独的梯度/参数诊断。

依赖：

```sh
pip install vector-quantize-pytorch
```

## Demo

`Music2LatentTrainingCallback` 默认写到 `${paths.output_dir}/demos`。每个样本对 raw model 和 EMA model 各保存一个 MP3；如果 wrapper 没有 EMA，只保存 raw：

```text
original -> continuous reconstruction -> discrete reconstruction
```

demo reconstruction 从 `sigma_max` 高斯噪声开始，按 `demo_denoising_steps` 调 `forward_generator()` 做 consistency decode；默认 `demo_denoising_steps: 1`。不要用 `forward_generator(..., x=clean_representation, sigma=sigma_min)` 做 demo，因为 EDM 边界条件下 `c_skip=1/c_out=0` 会把原始谱图原样返回，造成未训练也“完美重建”的假象。`demo_every` 默认 `1000`，逻辑是首次 batch 保存一次，之后每隔 `demo_every` 个 `global_step` 保存一次。文件名包含 `raw` 或 `ema`。

## 测试

```sh
python -m py_compile scripts/music2latent/audio.py scripts/music2latent/diffusion.py scripts/music2latent/models.py scripts/music2latent/wrapper.py
python -m pytest -q tests/music2latent
```
