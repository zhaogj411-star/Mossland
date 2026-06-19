# Music2Latent Local Transformer 改造方案

读取本文件用于恢复“用 `scripts/same` 的 local transformer 替换 Music2Latent 卷积结构，同时保留 skip connect，目标 `2*48000 -> 32*25`”这一设计任务。

## 目标

- 输入输出：48 kHz stereo waveform，1 秒窗口 `2*48000`。
- latent：`32` 维、`25 Hz`，即 1 秒输出 `[B,32,25]`。
- 训练口径：保留 Music2Latent 的 STFT real/imag 表示、EDM consistency high/low sigma pair、`c_skip/c_out/c_in`、latent pyramid decoder 和 U-Net skip connect。
- 主改动：用 `scripts/same.autoencoders.TransformerResamplingBlock` / local transformer 替换卷积 ResBlock/Downsample/Upsample 的主体建模。

## 推荐结构

- 新增独立模型类，不直接覆盖当前 `scripts/music2latent/models.py` 的默认 `Music2Latent`，建议命名 `Music2LatentLocalTransformer` 或放入新文件 `scripts/music2latent/local_transformer_models.py`。
- 保留 `AudioProcessor`，设置 `sample_rate=48000`、`hop=240`、`fac=4`。1 秒音频得到约 200 个 STFT frame；下采样 `8x` 后为 25 latent frames。
- Encoder 仍先处理 `[B,4,F,T]` STFT 表示，频率维需要先压入 token channel。可沿用原始频率下采样，最后 reshape 成 `[B,C*F,T]`，再接 SAME local transformer blocks 做时间建模和下采样。
- Encoder 时间下采样建议用 strides `[2,2,2]`，总倍率 `8`，把 `T=200` 变成 `25`。末端投影到 `[B,32,25]`，保留可选 hidden `[B,H,25]` 给 RVQ 或调试。
- Latent pyramid decoder 从 `[B,32,25]` 反向生成多尺度 2D feature pyramid，尺度必须对齐生成器 down path：例如 `T=200,100,50,25` 或按现有 `layers_list` 对齐。
- Generator/U-Net 保留原始 Music2Latent 的两类 skip：down path 自身 `skip_list`，以及每层注入 `pyramid_latents[i]`。只把每个 `ResBlock` 的内部局部建模替换为 local transformer block 或 hybrid block。

## 实施顺序

1. 先做 continuous-only：`quantizer_num_quantizers=0`，确认 `[B,2,48000] -> representation -> latent [B,32,25] -> reconstructed representation -> waveform` 形状全通。
2. 再接 wrapper 的 consistency loss，使用 10 条 overfit 配置验证 loss 可下降和 demo 可听。
3. 形状稳定后再恢复 RVQ 旁路；RVQ 应继续吃 detached hidden，不让离散分支改变主 consistency 路径。
4. 最后再考虑替换更多 2D 卷积。不要第一版就完全移除小卷积投影，因为频率局部结构和通道映射仍需要轻量卷积更稳。

## 风险点

- `TransformerResamplingBlock` 是 1D 序列模块，不能直接吃 `[B,C,F,T]`。必须先决定频率维是压成 channel、分 band token，还是保留轻量 2D frontend。
- 目标 `hop=240` 下 STFT frame 数与 25 Hz latent 正好对齐；如果换 hop 或裁剪长度，wrapper 必须裁到时间帧可被 8 整除。
- SAME decoder 是纯 autoencoder decoder；Music2Latent 的 consistency generator 还需要 noisy representation、sigma embedding、skip/pyramid 注入。不能直接拿 `SAMEDecoder` 替换整个 Music2Latent decoder。
- 保留 skip connect 的关键是保留 `pyramid_latents` 注入点和 U-Net `skip_list.pop()` 融合点，而不是保留原 ResBlock 类名。

## 当前落地

- 代码位置：`scripts/same_flow/`。
- `scripts/same_flow` 已按用户要求整理为自包含目录，运行时不 import `scripts/music2latent` 或 `scripts/same`；本地文件包含音频处理、RVQ、训练基类、SAME-style transformer 和 wrapper。
- `scripts/same_flow/models.py` 已改为显式实现 encoder、latent pyramid decoder 和 generator 层表；没有 `_replace_resblocks`，也不再先构造旧卷积 Music2Latent 后递归替换。
- 2026-06-18 进一步按用户要求全面向 SAME 靠拢：删除旧 `music2latent_base.py`，`models.py`/`transformer.py` 中不再使用旧 `DownsampleConv/UpsampleConv` 或可选 conv feed-forward 分支；down/up 采样用 `SameFlowResampling1d/2d`。
- 2026-06-18 按用户要求重新以 `scripts/same` 和 `scripts/music2latent` 为准校准：`SameFlowResampling1d` 改为按 SAME `TransformerResamplingBlock` 语义实现，包含 chunk padding、learned new tokens、sliding window、SAME mapping 和取输出 token；mapping 使用 parametrizations weight norm 以兼容 EMA。decode/reverse diffusion 恢复 Music2Latent 的随机 reverse-step 口径，不使用上一版自行假设的 deterministic refinement。`encode()` 非整除长度仍保留向上 padding，避免推理时裁掉尾巴。
- 首个配置：`scripts/configs/experiment/same-flow-overfit10.yaml`。
- 2026-06-18 已验证随机 smoke：`[1,2,48000] -> STFT [1,4,480,200] -> latent [1,32,25] -> generator output [1,4,480,200]`；`decode()` 输出 `[1,2,48000]`；随机 wrapper `training_step().backward()` finite。
