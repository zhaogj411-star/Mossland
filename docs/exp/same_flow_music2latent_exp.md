# Same 模块逐步替换 Music2Latent 实验

## 目标

以 `scripts/music2latent` 官方复现训练过程为基准，逐步把模型内部模块替换为 `scripts/same` 中的等价 SAME 模块，观察 loss 曲线、梯度和 head 激活变化。调试实现放在 `scripts/same_flow_debug`，当前阶段不再改 `scripts/same_flow` 正式实验目录。

## 固定约束

- 训练数据、STFT、裁剪长度、sigma sampling、consistency loss、optimizer、bf16 precision 和 wrapper 保持 `experiment=music2latent` 一致。
- 只在 `model._target_` 指向 debug hybrid 类时改变架构。
- 用户的 Music2Latent 长跑使用 0-3 卡；调试短跑使用 4-7 卡。
- 实验记录按窗口统计 TensorBoard `loss/total` 和 `grad_norm/before_clip`，避免用单步 loss 判断。
- 为支持变长音频编解码，时间维度替换必须保持局部性：不得使用 full-time attention。原 Music2Latent 卷积天然局部，替换为 SAME 模块时只能使用 local/sliding-window transformer。`denoiser_resblock_same_time` 现在会拒绝 `same_sliding_window=None`，避免误跑 full-time attention。

## 当前实现

新增 `scripts/same_flow_debug/music2latent_same_ablation.py`：

- `Music2LatentSAMEAblation`：继承官方 `Music2Latent`。
- `ablation_variant=official`：同一 target 下的原始 Music2Latent。
- `ablation_variant=denoiser_resblock_same_time`：只替换 denoising U-Net 的 `down_layers` / `up_layers` 中 20 个 `ResBlock`，encoder / decoder / audio processor / output head 保持官方 Music2Latent。
- `ablation_variant=denoiser_attention_same_freq`：只替换 denoising U-Net ResBlock 内 attention 子模块为 SAME `TransformerBlock` adapter，官方 Conv ResBlock 主体不变。
- `ablation_variant=denoiser_freq_resampling_same`：只替换 denoising U-Net 频率 resampling block 为 SAME resampling adapter。
- `ablation_variant=denoiser_attention_freq_resampling_same`：累计替换 attention 与 freq resampling；另有 `_down_only` / `_up_only` 用于只替换 downsample 或 upsample 方向。
- `SAMEResBlockAdapter`：使用 `scripts.same.transformer.TransformerBlock` 实现 residual branch。初始化保持 identity 行为：`cond_proj` 和 `out_proj` 零初始化，首步预测仍等价于官方零 residual head 的训练起点。

## 已完成检查

### 固定 batch 对齐

命令：

```bash
CUDA_VISIBLE_DEVICES=4 python -m scripts.same_flow_debug.compare_training_dynamics \
  --device cuda:0 \
  --batch-size 2 \
  --same-override model._target_=scripts.same_flow_debug.music2latent_same_ablation.Music2LatentSAMEAblation \
  --same-override model.audio_processor._target_=scripts.music2latent.audio.AudioProcessor \
  --same-override wrapper._target_=scripts.music2latent.wrapper.Music2LatnetTrainingWrapper \
  --same-override +model.ablation_variant=denoiser_resblock_same_time \
  --same-override +model.same_transformer_depth=1 \
  --same-override +model.same_dim_heads=64 \
  --same-override +model.same_ff_mult=2 \
  --same-override +model.same_sliding_window='[8,8]' \
  --output tmp/same_flow_debug/training_dynamics_music2latent_same_resblock_time.json
```

结果：

- Music2Latent loss: `0.2226602733`
- SAME-resblock candidate loss: `0.2226602733`
- Music2Latent grad norm: `0.4322574139`
- SAME-resblock candidate grad norm: `0.4216762483`
- `conv_out/input` RMS：官方 `0.635836`，候选 `0.631129`

结论：第一阶段替换的初始 loss、head 输入和梯度尺度是干净对齐的，适合进入短跑曲线比较。

## 当前短跑

已完成：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m scripts.train \
  experiment=music2latent \
  trainer.devices=4 \
  trainer.max_steps=1500 \
  callbacks.demo_callback.demo_every=999999 \
  logger.wandb=null \
  model._target_=scripts.same_flow_debug.music2latent_same_ablation.Music2LatentSAMEAblation \
  +model.ablation_variant=denoiser_resblock_same_time \
  +model.same_transformer_depth=1 \
  +model.same_dim_heads=64 \
  +model.same_ff_mult=2 \
  +model.same_sliding_window='[8,8]' \
  hydra.run.dir=logs/same-flow-baseline/runs/debug_music2latent_same_resblock_time_1500
```

模型参数量：`51.1M`，替换模块数：20 个 denoising ResBlock。

TensorBoard 窗口统计：

| run | 500-850 loss | 850-1000 loss | 1000-1250 loss | 1250-1500 loss | 1250-1500 grad |
| --- | ---: | ---: | ---: | ---: | ---: |
| official Music2Latent | `0.103678` | `0.094159` | `0.072380` | `0.047786` | `0.575770` |
| SAME ResBlock time | `0.099477` | `0.094117` | `0.089217` | `0.074835` | `0.502022` |

结论：该替换不是完全坏的，loss 能下降且梯度后段能放大；但 1000 step 后明显落后于官方，1250-1500 窗口 loss 均值比官方高约 `57%`。这说明在 Music2Latent 训练过程固定时，只把 denoising ResBlock 换成 SAME time-axis Transformer 会削弱官方 2D Conv ResBlock 的快速收敛能力。下一步应先拆 SAME block 内部因素，而不是继续大范围替换 encoder/decoder。

## 新 batch64 baseline 对齐

用户新启动的 Music2Latent baseline：

- Run: `logs/music2latent-official/runs/2026-06-19_16-48-51`
- 关键配置：`train_batch_size=64` per GPU、`trainer.devices=4`、`precision=bf16-mixed`、`wrapper.lr_warmup_steps=100`。
- 该 run 替代 `2026-06-19_16-47-16` 作为后续对比基准；`16-47-16` 仍是 `lr_warmup_steps=10000`。

### Attention-only SAME ablation

新增/修正 `ablation_variant=denoiser_attention_same_freq`：只替换 Music2Latent denoising U-Net ResBlock 内的 attention 子模块，保留官方 2D Conv ResBlock、encoder、decoder、audio processor、consistency loss 和 wrapper。adapter 输出显式恢复输入 dtype 并 `contiguous()`，避免 PPU bf16 后续 `Conv2d` 后端因布局/dtype 失败。

尝试对齐 batch64：

- `same_ff_mult=2,differential=true`：单卡 batch2 smoke 通过；4 卡 batch64 因 OOM/GET 后端错误失败。
- `same_ff_mult=1,differential=false`：参数量约 `60.0M`，接近官方 `57.7M`；4 卡 batch64 仍因显存/workspace 压力失败。
- 可运行设置：4 卡、`train_batch_size=48`、`bf16-mixed`、`lr_warmup_steps=100`、`same_ff_mult=1`、`same_differential=false`。

可运行命令：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m scripts.train \
  experiment=music2latent \
  trainer.devices=4 \
  trainer.max_steps=1500 \
  data.train_batch_size=48 \
  wrapper.lr_warmup_steps=100 \
  callbacks.demo_callback.demo_every=999999 \
  logger.wandb=null \
  model._target_=scripts.same_flow_debug.music2latent_same_ablation.Music2LatentSAMEAblation \
  +model.ablation_variant=denoiser_attention_same_freq \
  +model.same_transformer_depth=1 \
  +model.same_dim_heads=64 \
  +model.same_ff_mult=1 \
  +model.same_differential=false \
  +model.same_sliding_window='[8,8]' \
  hydra.run.dir=logs/same-flow-baseline/runs/debug_music2latent_same_attention_freq_bs48_warm100_ff1_nodiff_1500
```

该 run 手动停止在 step 547，足够判断曲线。TensorBoard 窗口统计：

| run | batch/GPU | 0-25 loss | 100-200 loss | 200-300 loss | 300-500 loss | 500-600 loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Music2Latent `16-48-51` | 64 | `0.100218` | `0.078251` | `0.040822` | `0.032658` | `0.028330` |
| SAME attention freq ff1 nodiff | 48 | `0.098409` | `0.078919` | `0.041774` | `0.031774` | `0.027862` |

对应 grad norm 均值：

| run | 0-25 grad | 100-200 grad | 200-300 grad | 300-500 grad | 500-600 grad |
| --- | ---: | ---: | ---: | ---: | ---: |
| Music2Latent `16-48-51` | `0.102684` | `0.241934` | `0.384000` | `0.340186` | `0.263021` |
| SAME attention freq ff1 nodiff | `0.099986` | `0.242317` | `0.318542` | `0.337790` | `0.435240` |

结论：在 Music2Latent 的训练过程、loss、数据和 wrapper 下，SAME attention-only 替换是能快速下降的，且 100-600 step 的 loss 窗口几乎贴近新官方 baseline。原 `scripts/same_flow` / `same_flow_baseline.yaml` 基本不降的问题不应归因于 SAME Transformer 完全不能学习，而应继续排查正式 same_flow 的训练目标、representation 到 output head 的尺度约定、输出 head 梯度聚合方式和模型输出残差语义。当前 candidate 不能完全用 batch64 复刻，原因是显存/workspace 压力；后续长跑可先用 batch48 或进一步减轻 attention adapter。

## Overfit10 perfect baseline 复刻

用户确认 `logs/music2latent-official/runs/2026-06-19_17-20-09` 的 10 条音频 demo 已完美过拟合，后续 SAME 替换以该 run 为 baseline。

关键配置：

- `trainer.devices=4`
- `data.train_batch_size=64`
- `trainer.precision=bf16-mixed`
- `wrapper.lr_warmup_steps=10`
- dataset 为 `tmp/music2latent_official_pt_10k/files.list` 中的小 overfit 集合；文件名为 `000000.pt` 到 `000010.pt`，`wc -l` 因末尾换行显示 10。

Baseline `loss/total`：

| run | last | 0-25 | 100-200 | 200-300 | 300-500 | 500-700 | 700-900 | 1500-1900 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Music2Latent `17-20-09` | step1901 `0.002392` | `0.104558` | `0.078914` | `0.045322` | `0.037001` | `0.031622` | `0.023426` | `0.004135` |

固定 batch 对齐结果，均使用 `same_ff_mult=1`、`same_differential=false`、`same_sliding_window=[8,8]`：

| candidate | output json | official loss | candidate loss | official grad | candidate grad | candidate `conv_out/input` RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attention only | `tmp/same_flow_debug/overfit10_attention_fixed_batch.json` | `0.1537591666` | `0.1537591666` | `0.312643` | `0.306193` | `0.645644` |
| freq resampling only | `tmp/same_flow_debug/overfit10_freq_resampling_fixed_batch.json` | `0.1537591666` | `0.1537591666` | `0.312643` | `0.306825` | `0.646486` |
| attention + freq resampling | `tmp/same_flow_debug/overfit10_attention_freq_resampling_fixed_batch.json` | `0.1537591666` | `0.1537591666` | `0.312643` | `0.307016` | `0.646506` |

结论：这些替换的首步训练语义是干净的；loss、目标、sigma/noise、head 梯度和 head 输入尺度没有明显断裂。

### Batch32 accumulate-2 口径

4 卡 batch64 在 attention adapter 后续 `Conv2d` 处会触发 PPU 后端错误 `GET was unable to find an engine to execute this computation`，更像显存/workspace 后端问题，不是训练目标错误。可运行口径改为每卡 batch32、`trainer.accumulate_grad_batches=2`，保持 4 卡 global effective batch 256，与 baseline 等效。

Attention-only 短跑：

- Run: `logs/same-flow-baseline/runs/debug_overfit10_music2latent_same_attention_freq_bs32_acc2_warm10_700`
- 手动停止：step443
- 窗口：0-25 `0.104912`、100-200 `0.079114`、200-300 `0.046191`、300-443 `0.036596`

与 baseline 对比，前 300 step 基本同趋势，300-443 窗口也贴近 baseline 300-500 的 `0.037001`。

### 当前并行实验

按用户要求占满 8 卡同步跑两个替换实验，均为 4 卡、每卡 batch32、`accumulate_grad_batches=2`、`max_steps=900`、`demo_every=100`：

| GPUs | run | variant | latest | 0-25 | 100-200 | 200-300 | checkpoint/demo |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0-3 | `logs/same-flow-baseline/runs/debug_overfit10_music2latent_same_attention_freqresamp_bs32_acc2_warm10_900` | `denoiser_attention_freq_resampling_same` | step279 `0.050482` | `0.104912` | `0.081304` | `0.046083`（n=80） | 100/200 ckpt，1/101/201 demo |
| 4-7 | `logs/same-flow-baseline/runs/debug_overfit10_music2latent_same_freqresamp_bs32_acc2_warm10_900` | `denoiser_freq_resampling_same` | step300 `0.028404` | `0.104913` | `0.081219` | `0.044420`（n=100） | 100/200 ckpt，1/101/201 demo |

结论：freq-resampling-only 已在 200-300 窗口达到 baseline 同级甚至略低；attention+freq-resampling 也保持正确下降趋势，只是目前窗口略慢。继续等 500/700/900 窗口和 demo 试听，不需要因为 transformer 慢一些就判失败。

### ResBlock 主体替换约束

ResBlock 主体替换沿时间维度 token 化：`SAMEResBlockAdapter` 对 2D feature 使用 `[batch * freq, frames, channels]` token 序列，因此 `same_sliding_window` 是时间局部窗口。为了保持变长音频编解码能力，不能把该窗口设为 `null`。

用户后续明确指出：不要用 activation checkpointing 等 trick 掩盖问题，应回到模型行为本身；原 ResBlock 是两层 `3x3` Conv2d，单层时间半径为 1，频率/时间都天然局部。因此 ResBlock 替换的 local transformer 窗口应优先按卷积等价语义使用 `[1,1]`，而不是继续做 `[8,8]` vs `[16,16]` 这类超参搜索。

### 模块行为审计

用户要求每个 SAME 替换模块都必须和原 Music2Latent 在行为上接近，包括感受野、计算量和 activation 行为；差异很大的模块不能作为正式替换口径，只能作为诊断。

- 原 `ResBlock` 是 2 层 `3x3` Conv2d，单层局部足迹为 `3x3`，两层 residual branch 有效局部足迹约 `5x5`，且频率和时间同时局部混合。旧 `SAMEResBlockAdapter` 把 `[B,C,F,T]` 变为 `[B*F,T,C]`，只沿时间做局部 attention，完全没有单层频率局部混合；即使 `same_sliding_window=[1,1]`，也只是时间轴半径 1，行为不等价。该分支后续只保留为诊断，不作为最终路线。
- `SAMELocal2DResBlockAdapter` 更接近原 ResBlock：直接在 `[B,C,F,T]` 上做 `k=3` 的 2D local attention，两层后有效足迹约 `5x5`，并保留 norm、SiLU、time embedding、residual、dropout 和 optional frequency attention。粗略 MAC（忽略 norm/softmax）在主要 stage 约为原两层 dense `3x3` Conv2d 的 `0.45x~0.48x`；batch2 activation peak `5.921GB` vs 官方 `3.881GB`，约 `1.5x`，明显高于 Conv2d 但比旧 time-token 的 `7.571GB` 健康。它仍不是严格 Conv2d：attention 是 content-adaptive 且 softmax 归一化，边界 padding 会作为候选 key/value 参与归一化；因此是目前较合理的局部动态卷积候选，而非 bit-equivalent 替换。
- 原 `Attention(use_2d=True)` 是每个 time frame 内沿频率的 full MHA，形状为 `[B*T,F,C]`，且没有 FFN。`SAMEAttentionAdapter` 的 token 轴相同，也是 `[B*T,F,C]`，但默认 `same_sliding_window=[8,8]` 会把原 full-frequency attention 改成局部频率 attention，并且 `TransformerBlock` 额外包含 FFN。该分支训练曲线正常，但若严格追求原模块行为，应新增 no-FFN / full-frequency 或按明确局部频率窗口重写的 attention adapter。
- 原 `DownsampleFreqConv` / `UpsampleFreqConv` 是频率维度 `(5,1)` 卷积，stride 或 nearest-upsample factor 为 4，时间维度不混合。当前 `SAMEFreqDownsampleAdapter` / `SAMEFreqUpsampleAdapter` 用 `TransformerResamplingBlock` 处理频率 token；默认 window `[8,8]` 在 stride=4 resampling 里对应远大于 `(5,1)` 的频率感受野，且 activation 行为和 Conv2d 差异较大。该模块虽然固定 batch 和短跑趋势正常，但行为审计下不能当作卷积等价替换；后续应改成局部频率 `(5,1)` 足迹的 attention/resampling，或暂缓替换 freq resampling。
- 已实现诊断版 `SAMELocalFreqDownsampleAdapter` / `SAMELocalFreqUpsampleAdapter`，每个 downsample 输出频点只看输入中心 `4*j` 附近 5 个频点，upsample 先 nearest `x4` 后看频率 5 点邻域，时间维度不混合。固定 batch loss 对齐干净：`tmp/same_flow_debug/overfit10_freq_resampling_local_k5_fixed_batch.json` 中 official/candidate loss 都是 `0.1537591666`，grad `0.312643` vs `0.304122`。但 4 卡、每卡 batch64 的短跑 `debug_overfit10_music2latent_same_freqlocal_k5_bs64_warm10_30` 在第 0 step 前向 OOM，单进程约占 `95GB`，失败点是 local freq attention 的 Q/K/V padding activation。结论：它的感受野对齐了，但 activation 行为仍和 Conv2d 差太远，不能作为正式替换；`same_freq_resampling_kind=local` 只保留为诊断显式 opt-in，默认仍是旧 transformer 诊断路径。

当前正式推进顺序应调整为：先以 `SAMELocal2DResBlockAdapter(k=3)` 替换 ResBlock 并优化实现；旧 time-token ResBlock 和当前 TransformerResamplingBlock 频率 resampling 只做诊断，不再作为最终替换路线。

### Step 速度基准

按 TensorBoard scalar `loss/total` 的 wall_time 差分估计训练 step 速度：

| run | batch/GPU | effective batch | median sec/step | trimmed it/s | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| official `logs/music2latent-official/runs/2026-06-19_17-20-09` | 64 | 256 | `1.435` | `0.697` | baseline，可 batch64 |
| `debug_overfit10_music2latent_same_attention_freq_bs32_acc2_warm10_700` | 32 | 256 | `1.697` | `0.589` | attention-only，约慢 `18%` |
| `debug_overfit10_music2latent_same_freqresamp_bs32_acc2_warm10_900` | 32 | 256 | `3.010` | `0.332` | 旧 TransformerResamplingBlock freq，约 `2.1x` 慢 |
| `debug_overfit10_music2latent_same_attention_freqresamp_bs32_acc2_warm10_900` | 32 | 256 | `3.250` | `0.308` | attention + 旧 freq，约 `2.3x` 慢 |
| `debug_overfit10_music2latent_same_resblock_time_ff1_nodiff_win8_bs32_acc2_warm10_700` | 32 | 256 | `3.022` | `0.331` | 旧 time-token ResBlock，行为不等价 |
| `debug_overfit10_music2latent_same_resblock_local2d_k3_bs32_acc2_warm10_300` | 32 | 256 | `4.879` | `0.205` | 2D local ResBlock，当前 Python 循环实现慢 |

该速度对比再次说明：真正可接受的替换不能只看 loss 趋势，至少应接近官方 batch64 的显存和 step 速度。当前 attention-only 速度尚可；freq resampling 和 ResBlock local attention 都需要更接近 Conv2d 的实现，不能用 batch/accumulate 降低来掩盖。

已误启动过一条 full-time 对照：

- Run: `logs/same-flow-baseline/runs/debug_overfit10_music2latent_same_resblock_time_ff1_nodiff_fullattn_bs32_acc2_warm10_700`
- 状态：已按用户要求停止，不作为有效实验。

当前 ResBlock 排查结论：

- `[8,8]` time-only run `debug_overfit10_music2latent_same_resblock_time_ff1_nodiff_win8_bs32_acc2_warm10_700` 到 step240，窗口 100-200 `0.096185`、200-240 `0.075721`，明显慢于 baseline 100-200 `0.078914`、200-300 `0.045322`。time-only 只沿时间每个频点独立 attention，缺少原 2D Conv 的频率局部混合。
- 曾添加 `denoiser_resblock_same_time_freq` 轴向 time+freq local variant，但 batch32 OOM，batch16/acc4 跑到 step33 后长时间不继续写 event；已停止，不作为有效方向。
- 新增 `denoiser_resblock_freq_resampling_same_time`：累计 SAME freq resampling 与 local-time ResBlock 替换。固定 batch 对齐干净，output 为 `tmp/same_flow_debug/overfit10_resblock_freqresamp_time_ff1_nodiff_fixed_batch.json`。
- 显存异常不是参数量：全量 ResBlock+freq 只有约 `46.9M` 参数，小于官方 `57.7M`，但 batch2 peak `10.94GB`。窗口 `[8,8]` 改 `[1,1]` 基本不降显存；主因是当前 TransformerBlock 为 `[B*freq, frames, channels]` 保存完整 Q/K/V activation，高分辨率 `freq=1024` stage 代价远大于 Conv2d。
- 新增 `same_resblock_min_channels` 做 stage-wise 替换。`same_resblock_min_channels=256` 只替换 12 个低分辨率 ResBlock，batch2 peak 降到 `7.743GB`；`128` 替换 16 个，peak `8.813GB`；全量 20 个 peak `10.943GB`。`min_channels=256, window=[1,1]` 固定 batch 对齐干净：official/candidate loss 都 `0.1537591666`，grad `0.312643` vs `0.312671`。
- 曾启动 batch32/acc2、`min_channels=256, window=[1,1]` 300-step run `debug_overfit10_music2latent_same_resblock_freqresamp_time_minch256_win1_bs32_acc2_warm10_300`；能跑但每卡仍达 `90GB+`，已停止，仅作为显存诊断。
- 新增 2D local attention ResBlock adapter：`Local2DAttentionConv` 直接在 `[B,C,F,T]` 上让每个位置 attend `3x3` 邻域，`SAMELocal2DResBlockAdapter` 保留 Music2Latent ResBlock 的 norm/activation/time embedding/residual/optional attention 结构，第二个 local-attn 输出 zero-init。Variant：`denoiser_resblock_same_local2d`、`denoiser_resblock_freq_resampling_same_local2d`。该方向比旧 `[B*freq,T,C]` time-token adapter 更接近 Conv2d 行为。
- 固定 batch `denoiser_resblock_same_local2d,k=3`：output `tmp/same_flow_debug/overfit10_resblock_local2d_k3_fixed_batch.json`，official/candidate loss 都 `0.1537591666`，grad `0.312643` vs `0.307322`，head RMS 正常。
- batch2 peak 显存：official `3.881GB`，旧 time-token `window=[1,1]` `7.571GB`，local2d k3 `5.921GB`，local2d k3 + SAME freq resampling `9.311GB`。说明 2D local attention 明显缓解高分辨率 ResBlock 的 QKV activation 问题；freq resampling 仍是单独重项。
- 短跑 `logs/same-flow-baseline/runs/debug_overfit10_music2latent_same_resblock_local2d_k3_bs32_acc2_warm10_300` 到约 step104 手动停止，TensorBoard 只 flush 到 step52：0-25 `0.104915`、25-50 `0.100696`，贴近 baseline 0-25 `0.104558`、25-50 `0.101856`。batch32/acc2 显存约 `73GB/GPU`，速度约 `0.40 it/s`。后续若继续该方向，应优化 9 邻域循环实现或跑到 200/300 step 看下降窗口。

后续所有时间轴 SAME 替换都必须显式使用 local/sliding-window attention。

## 对比基准

旧官方 Music2Latent 长跑：

- Run: `logs/music2latent-official/runs/2026-06-19_15-22-07`
- 关键参考窗口：`loss/total` 在 1000-1250 step 约 `0.075`，1250-1500 step 约 `0.048`。

## 下一步

### SAME-shape 1D patch 新架构线

按用户要求新增第二条线：不再做逐模块等价替换，而是把 Music2Latent 输入调整成接近 SAME 的 1D patch/token 形态，让 encoder、decoder、denoiser 都沿时间 1D SAME resampling attention 工作。

配置：`scripts/configs/experiment/music2latent-same-1d-patch.yaml`

关键参数：

- 数据、sigma sampling、consistency loss、optimizer、bf16、EMA、warmup=10、batch/GPU=64 对齐 `logs/music2latent-official/runs/2026-06-19_17-20-09`。
- 当前默认 STFT 改为 `hop=64, fac=4, center_pad=false`；这是为了按用户判断先降低每个 frame 的频率宽度、增加 time tokens，不追求一开始就达到 Music2Latent 的高压缩率。
- 当前默认 representation 为约 `[B,2,128,528]`，每个 frame 只有 `2*128=256` 个频谱值；4 次 time stride 后 latent 为 `[B,64,33]`，单样本 latent floats `2112`，约为官方 `[B,64,8]` 的 `4.1x`。
- `SpectralFrameAdapter(freq_patch_size=16, patch_channels=32, frame_channels=512)`：每个 STFT time frame 打包成一个 512-d frame token，频率维度只在 adapter 内压入 channel，不再做 2D U-Net。
- `freq_downsample_list=[0,0,0,0]` 在 `SameFlowConsistencyAutoencoder` 中表示 4 次 stride-2 time resampling；time-only SAME/local attention 约束保持不变。
- 参数量 `54.10M`，接近官方 `57.66M`。

验证：

```bash
python -m scripts.train experiment=music2latent-same-1d-patch \
  trainer.devices=1 trainer.strategy=auto trainer.max_steps=1 trainer.max_epochs=1 \
  data.train_batch_size=2 data.num_workers=0 callbacks.demo_callback.demo_every=999999 \
  logger.wandb=null trainer.enable_model_summary=false trainer.num_sanity_val_steps=0 \
  extras.print_config=false hydra.run.dir=tmp/same_flow_debug/music2latent_same_1d_patch_smoke
```

旧 `hop=256` 配置通过，单卡 1 step loss `0.0527`。当前 `hop=64` 默认配置 instantiate 通过：`hop=64`、`freq_bins=128`、参数量 `54.10M`。

4 卡 batch64 300-step 短跑命令示例：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m scripts.train \
  experiment=music2latent-same-1d-patch trainer.devices=4 trainer.max_steps=300 \
  trainer.max_epochs=1 data.train_batch_size=64 data.num_workers=8 \
  logger.wandb=null trainer.enable_model_summary=false trainer.num_sanity_val_steps=0 \
  extras.print_config=false \
  hydra.run.dir=logs/music2latent-same-1d-patch/runs/debug_overfit10_same1dpatch_bs64_warm10_300
```

结果：

| run | batch/GPU | 0-25 | 25-50 | 50-100 | 100-200 | 200-300 | 700-900 | 900-1200 | 1200-1500 | 1500-1800 | speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official `17-20-09` | 64 | `0.104558` | `0.101856` | `0.102106` | `0.078914` | `0.045322` | `0.023426` | `0.010833` | `0.006741` | `0.004511` | `0.697 it/s` |
| SAME 1D patch `hop=256` | 64 | `0.096572` | `0.094410` | `0.096263` | `0.094728` | `0.092776` | - | - | - | - | `2.567 it/s` |
| SAME 1D patch `hop=128` | 64 | `0.091021` | `0.088737` | `0.089370` | `0.088076` | `0.086346` | - | - | - | - | `~2.2 it/s` |
| SAME 1D patch `hop=64` | 64 | `0.085850` | `0.083146` | `0.082345` | `0.080347` | `0.077436` | `0.070986` | `0.071273` | `0.069463` | `0.069127` | `~2.0 it/s` |

长跑使用 `debug_overfit10_same1dpatch_hop64_bs64_warm10_2000_resume625` 从 625-step checkpoint 续训；截至 step1855，hop64 已基本平台在 `0.069` 左右，没有进入官方 0.01 量级。

结论：用户指出的“每帧频率维直接压得太狠”是有效方向。`hop=256 -> 128 -> 64` 时，频率 bins 降低、time/latent tokens 增加，300-step loss 单调改善，且 hop64 显存仍健康（约 `16GB/GPU`）。但长跑证明当前 1D patch 架构训不出 Music2Latent 级别的重建，不是单纯训练步数不足。

当前更可疑的结构问题：

- 用户指出不能简单说“频率没有交互”：`SpectralFrameAdapter` 的 frame projection 和 `ZeroTokenProjection(input_channels -> data_channels*freq_bins)` 都是频率全连接，最后一次性输出整列频谱本身不是必然错误。
- 更准确的问题是 token 粒度和信息分配：每个 time token 同时承担整帧频谱细节、skip、噪声条件和 latent 条件；Music2Latent 则把同样信息摊在 `[F,T]` 网格的很多 feature cell 上，通过局部共享算子逐层更新。
- `output_residual_scale=64` 只能补 frame-linear head 初始梯度尺度，不应作为最终设计依赖。下一版应让每个 token 承载更少频谱信息，例如保留 frequency patch/grid tokens 到 denoiser，并使用 patch-wise/local 2D residual head，而不是继续优先调 `hop/window`。

## 下一步

1. 模块替换线：只保留 attention-only 作为行为/速度都相对合理的替换；ResBlock 继续优化 2D local 实现，freq resampling 暂缓。
2. 1D patch 新架构线：`hop=64` 长跑已平台，下一步重做频率局部 token/输出 residual head，再回到 overfit10 验证。
