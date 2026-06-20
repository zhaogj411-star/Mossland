# Mossland Codec RVQ Bottleneck Design

> **Superseded:** Do not implement this chunk-summary RVQ design. It was replaced by `docs/superpowers/specs/2026-06-17-mossland-codec-time-aligned-rvq-design.md` after the user clarified that the current learned-query Transformer summary is the source of the time-alignment problem.

## 背景

用户希望把 `scripts/mossland-codec` 的 FSQ 离散瓶颈替换为 RVQ，并让 RVQ 可以像现有 FSQ 一样参与训练。现有模型每个 `spec_length` chunk 产生 `num_latents=128` 个 bottleneck latent，每个 latent 是 `len(fsq_levels)=4` 维；因此一个 chunk 的离散瓶颈总维度是 `128 * 4 = 512`。

当前主训练配置的 Transformer hidden dim 是 `768`，但 RVQ 作用的不是 hidden dim，而是 encoder 输出后的 bottleneck 表征。

## 目标

- 删除 Mossland codec 本地 FSQ 实现和 FSQ 专属训练命名。
- 新增 RVQ bottleneck：按 chunk 把 `[128, 4]` 展平成单个 512 维向量，在 512 维上做 residual vector quantization。
- 默认 RVQ 参数为 `codebook_size=1024`、最大 `num_quantizers=256`。
- 像 DAC 一样支持动态码本数：训练、encode 和 decode 都可以指定实际使用的 `n_quantizers <= 256`；未指定时使用全部 256 层。
- 保留连续 latent 训练路径，并让 RVQ 在预热后以可配置概率进入 decoder conditioning。
- 保持现有 decoder、pre-decoder、parallel decode 和 autoregressive decode 的内部 latent 形状不变，降低改造风险。

## 码率

现有 FSQ 每 chunk 的理论 bit 数是：

```text
128 latents/chunk * 4 dims/latent * log2(11) = 1771.23 bits/chunk
```

新的 RVQ 最大配置为：

```text
256 quantizers/chunk * log2(1024) = 2560 bits/chunk
```

因此 RVQ 使用全部 256 层时码率约为当前 FSQ 的 `1.45x`。在当前 `spec_length=32`、`hop=1024`、`sample_rate=44100`、128x 时间压缩口径下，FSQ 约 `2.38 kbps`，RVQ 256 层约 `3.44 kbps`。若指定 `n_quantizers`，码率按 `n_quantizers * 10 bits/chunk` 线性变化；例如 `178` 层约接近原 FSQ 码率。默认最大 256 层是用户确认的质量优先取舍。

## 张量接口

模型内部继续使用现有 latent layout：

```text
encoder output:        [B, chunks * 128, 4]
flatten per chunk:     [B, chunks, 512]
RVQ quantize:          [B, chunks, 512]
RVQ codes:             [B, chunks, n_quantizers]
unflatten for decoder: [B, chunks * 128, 4]
```

不要第一版把 decoder 改成只接收 `[B, chunks, 512]`。那会牵动 `pre_decoder_forward()`、mask embedding、frontend feature upsampling、parallel decode 分块和 autoregressive decode 状态管理，风险高于收益。

RVQ 模块内部保留 256 层最大码本。`forward(..., n_quantizers=None)` 默认使用全部层；传入较小 `n_quantizers` 时只使用前 N 层。`codes_to_latents(codes)` 根据 `codes.shape[-1]` 自动推断使用层数，因此 `[B, chunks, 64]`、`[B, chunks, 128]`、`[B, chunks, 256]` 等 code tensor 都可解码。

## 模块设计

新增 `scripts/mossland-codec/quantization.py`，包含：

- `LatentTanhBound`：替代 FSQ 的 continuous bound。即使不量化，也把 encoder bottleneck 限制在稳定范围内，例如 `tanh(x) * (1 - eps)`。
- `VectorQuantizer`：单层码本，输入 `[B, T, D]`，输出 nearest code、quantized vector、indices、commitment/codebook loss 和 usage 指标。
- `ResidualVectorQuantizer`：按 residual 逐层调用 `VectorQuantizer`，初始化最大 `num_quantizers=256`、`codebook_size=1024`，forward 时支持 `n_quantizers` 动态层数。
- `MosslandRVQBottleneck`：Mossland 专用 wrapper，负责 `[B, chunks * 128, 4]` 与 `[B, chunks, 512]` 的 reshape，以及 `latents_to_codes(..., n_quantizers=None)` / `codes_to_latents()`。

参考 Descript DAC 的 residual VQ 结构，但不照搬其训练目标。DAC 量化模块参考：https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py

## 训练策略

推荐“旁路预热 + 混入 decoder”的两阶段策略：

1. RVQ warmup 阶段：
   - decoder 只使用 continuous latent；
   - RVQ 旁路接收 `z_cont.detach()`；
   - 优化 RVQ 自身的 `rvq_recon_loss + rvq_codebook_loss`；
   - 不让 RVQ loss 更新 encoder，避免随机码本早期扰乱连续表征。

2. RVQ active 阶段：
   - 每个 step 按 `rvq_path_dropout_prob` 选择 continuous 或 RVQ latent 给 decoder；
   - 语义沿用现有 `fsq_dropout_prob`：概率越高，continuous 路径越常用；
   - RVQ active 时可用 `rvq_active_quantizers` 固定层数，或用 `rvq_min_active_quantizers` 到最大层数之间随机采样训练层数；默认不采样，使用全部 256 层；
   - RVQ 继续通过辅助 loss 学习还原 `z_cont.detach()`；
   - 默认不使用或极小使用 commitment loss 反向更新 encoder，除非后续实验证明需要。

训练总 loss：

```text
loss = consistency_loss + rvq_loss_weight * rvq_loss
rvq_loss = rvq_recon_loss + codebook_loss_weight * rvq_codebook_loss
```

默认建议：

```text
rvq_loss_weight = 1.0
codebook_loss_weight = 1.0
commitment_loss_weight = 0.0
rvq_warmup_steps = 10000
rvq_path_dropout_prob = 0.6
rvq_active_quantizers = null  # null means use max 256
rvq_min_active_quantizers = null  # null means no random layer-count sampling
```

这些默认值应进入 Hydra config，不能硬编码在训练 wrapper 内。

## 推理接口

`EncoderDecoder.encode(..., discrete=False)` 继续返回连续 latent，保留现有 `desired_channels` reshape 语义。

`EncoderDecoder.encode(..., discrete=True, n_quantizers=None)` 返回 RVQ codes，形状为：

```text
[chunks, n_quantizers]
```

批处理内部可使用 `[B, chunks, n_quantizers]`。`decode()` 若收到 integer codes，应调用 `quantizer.codes_to_latents()`，由 code tensor 最后一维自动推断层数并还原为 `[B, chunks * 128, 4]`，再走现有 decoder。

demo 输出不再拆成两个文件。`infer_demo.py` 和训练 demo callback 每个样本保存一个 concat 对比音频，顺序为 `src -> target -> rvq/discrete decode -> continuous decode`，中间按现有静音间隔分隔，用于直接比较离散编码和连续编码解码结果。

## 兼容性边界

- 新配置不兼容旧 FSQ checkpoint 的 FSQ 参数；加载旧 checkpoint 时应要求 `strict=False` 或只在新实验中使用。
- `scripts/codicodec/` 是独立参考实现，不跟随本次改造。
- A2A 配置当前 `use_fsq=false`，不应被 RVQ 改造影响。
- 代码命名应从 `fsq_*` 迁移到 `quantizer_*` 或 `rvq_*`，但旧 checkpoint 相关评估表不需要重写历史文字。

## 测试计划

- 单元测试 RVQ shape：`[B, 128, 4]` 与 `[B, 1, 512]` 往返、multi-chunk 往返、不同 `n_quantizers` 的 codes decode 往返。
- 单元测试 `EncoderDecoder.encode(discrete=True, n_quantizers=N)` 返回 integer RVQ codes，`decode(codes)` 能按 codes 最后一维进入原 decoder latent shape。
- 训练 wrapper smoke：warmup 阶段 decoder 用 continuous，active 阶段可控随机选择 RVQ path，并记录 `loss/rvq_*`、`latent/rvq_active`、code usage。
- demo callback 测试：每个样本只保存一个 concat wav，内容顺序包含 source、target、RVQ/discrete decode、continuous decode。
- 配置测试：`mossland-codec.yaml` 不再包含 `fsq_levels`/`fsq_dropout_prob`，改为 RVQ 参数。
- 现有 codec import、task、attention 和 eval benchmark 轻量测试继续通过。

## 非目标

- 第一版不改 decoder 的 token layout。
- 第一版不追求与 DAC bitstream 或 checkpoint 格式兼容。
- 第一版不重新评估历史 FSQ checkpoint，也不修改已落表的历史指标。
