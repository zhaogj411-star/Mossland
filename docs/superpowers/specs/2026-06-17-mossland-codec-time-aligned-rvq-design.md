# Mossland Codec Time-Aligned RVQ Design

## 背景

旧 RVQ 方案把一个 `spec_length` chunk 内的 `[128, 4]` bottleneck latent 展平成 512 维 summary embedding，再做 RVQ。用户指出这不能解决长音频 decode 的边界问题：当前无实质时间顺序的根因不是 reshape，而是 encoder Transformer 使用 learned latent queries 对 chunk 做全局 summary。仅把 summary latent 重排成 time tokens 仍然不是时间对齐。

因此旧文档 `2026-06-17-mossland-codec-rvq-design.md` 和旧实施计划 `2026-06-17-mossland-codec-rvq-implementation.md` 已废弃，不应执行。

## 目标

- 删除主 codec 的 FSQ 离散瓶颈，新增 RVQ。
- 不再把整 chunk 聚合成单个 512-D summary embedding。
- encoder 输出必须是由固定时间位置锚定的 latent token 序列：`[B, T_token, D_token]`。
- RVQ 在 token 维上做：每个时间 token 独立 residual vector quantization。
- 使用 local Transformer，而不是 learned-query global summary Transformer，避免每个 latent token 无边界地混合整个 chunk。
- 推理 API 面向连续 token 序列；decode 内部可以按窗口分组，但用户不应被迫复原 encode 时的原始 chunk 边界。
- 最终目标表示：`2 * 48000 Hz` stereo audio -> `25 Hz` latent token rate。连续表征为 `32` 维/token；离散表征为每 token `16` 层 RVQ codebook。

## 推荐架构

### Encoder

保留 STFT representation 和 frontend downsampling，但替换 learned latent query bottleneck：

```text
audio
-> STFT representation [B, C, F, T]
-> frontend_encoder_down preserving time grid
-> time-token projector [B, T_token, D_token]
-> LocalTransformerEncoder [B, T_token, D_token]
-> bounded continuous tokens [B, T_token, D_token]
```

`T_token` 必须和时间位置有固定映射。目标 token rate 是 `25 Hz`；在 `sample_rate=48000` 时，每个 token 对应 `1920` waveform samples，约 `40 ms`。

Local Transformer 建议使用双向局部窗口作为第一版，以质量优先；如果未来需要 streaming，再增加 causal/local causal 选项。关键约束是每个 token 只能看附近固定窗口，而不是 attend 整个长序列或整个 chunk。实现上也必须避免 full `T x T` attention 后再套 mask；应使用滑窗/块稀疏 attention，使注意力 logits 规模随 `T * window` 增长。当前实现优先走 `flash_attn_func(window_size=(w,w))`，仅在 CPU、float32、head_dim 不适配或缺少 `flash-attn` 时回退到 PyTorch `unfold` 滑窗。

### RVQ

RVQ 输入改为：

```text
continuous tokens: [B, T_token, 32]
rvq codes:         [B, T_token, n_quantizers]
rvq latents:       [B, T_token, 32]
```

仍使用 `codebook_size=1024`，并支持 DAC 式动态 `n_quantizers`。默认最大层数为：

```text
rvq_max_quantizers_per_token = 16
```

用满时：

```text
25 tokens/s * 16 quantizers/token * log2(1024) = 4000 bps
```

因此 16 层、1024 词表、25Hz 的离散码率约为 `4 kbps`。若指定 `n_quantizers < 16`，码率按 token 线性下降。连续对外表示为 `32 * 25Hz` float token sequence。

### Decoder

第一版可以保留 diffusion decoder 的固定窗口生成方式，但输入接口应变为 25Hz token sequence。内部 adapter 负责把连续的 time tokens 转换成 decoder 所需 conditioning：

```text
RVQ/continuous tokens [B, T_token, D_token]
-> window consecutive 25Hz tokens for decoder
-> decoder conditioning/features
-> waveform/repr window
-> stitch/parallel decode
```

这样 decode 仍可能内部按窗口运行，但窗口由 token stride 决定，可以从任意长 token 序列切连续窗口，不要求用户按原 encode chunk 边界保存和恢复 512-D summary chunks。

长期更干净的方案是让 `pre_decoder_forward()` 直接从 `[B, T_token, D_token]` 生成 decoder-aligned multiscale features，而不是先扩回旧 `[128,4]` latent slots。

## 训练策略

保留“continuous path + RVQ path”的双路径思想：

- warmup 阶段：decoder 使用 continuous time tokens；RVQ 旁路学习还原 `z_cont.detach()`。
- active 阶段：按 `rvq_path_dropout_prob` 在 continuous 和 RVQ token path 之间切换。
- 支持 `rvq_active_quantizers` 或 `rvq_min_active_quantizers` 动态控制每个 token 使用多少 RVQ 层。
- commitment loss 默认 `0.0` 或很小，避免早期把 encoder 拉向不稳定码本。

## Demo

训练 demo 和 `infer_demo.py` 每个样本只保存一个 concat wav：

```text
src -> target -> rvq/discrete decode -> continuous decode
```

中间使用现有静音间隔分隔。

## 需要重写的旧计划部分

旧计划中的以下内容不能执行：

- `MosslandRVQBottleneck.embedding_dim = num_latents * bottleneck_channels = 512`
- 每个 chunk 一个 RVQ code sequence `[B, chunks, n_quantizers]`
- 保持 learned latent query encoder 不变
- 仅通过 reshape `[128,4]` 来制造 time tokens

新的实施计划必须先改 encoder latent interface，再接 RVQ。

## 开放问题

- `D_token` 已定为 `32`，对应连续表征 `32 * 25Hz`。
- local window 大小：第一版可用 8 到 16 个 time tokens 的双向窗口。
- decoder adapter 是先扩回旧 latent slots，还是直接生成 multiscale features；前者改动小，后者架构更干净。
- 是否保留旧 `num_latents=128` 参数，或替换为 `latent_rate_hz=25`、`token_dim=32`、`rvq_max_quantizers_per_token=16` 等新参数。
