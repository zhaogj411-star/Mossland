# SAME Flow 实验记录

本文记录 `scripts/same_flow/` 用 SAME 风格 local transformer 复现 Music2Latent 一致性生成式重建模型的实验过程、结构调整、命令和结论。

## 目标

- 训练一个一致性生成式重建模型：`wav -> latent -> wav`。
- 保留 Music2Latent 的 consistency 训练口径、skip/pyramid 条件路径和 RVQ 扩展点。
- 主体架构向 SAME 靠拢：以 1D token 序列做 local transformer/resampling。
- 目标码形：1 秒 48 kHz stereo 时 `2*48000 -> 32*25`；当前 overfit10 配置用 2 秒窗口，因此是 `2*96000 -> 32*50`。

## 当前实现

- `Music2LatentSameFlow` 已切为 SAME-L 风格 1D frame-token 模型。
- STFT 表示仍是 Music2Latent 形式 `[B,4,480,T]`。
- `SpectralFrameAdapter` 先按 32-bin frequency patch 压成 `[B,512,T]`，避免直接让 SAME 主干吃 `in_channels=1920`。
- encoder、decoder、denoiser 和 skip/pyramid 都沿时间轴使用 SAME-style 1D local transformer/resampling。
- `freq_downsample_list=[1,0,0,0]` 在该版本中表示第一层不降时间、后三层 stride 2，总时间下采样 8。
- 当前 `same-flow-overfit10.yaml` 是 continuous-only，`quantizer_num_quantizers=0`。

## 固定评估

新增脚本：

```sh
python -m scripts.same_flow.eval_checkpoint \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_eval_step0 \
  --max-items 10 \
  --denoising-steps 4 \
  --device cuda:0
```

脚本输出每条样本的 `source.wav`、`reconstruct_{raw,ema}.wav` 和 `results.json`，指标包含 `corr`、`snr_db`、`lsd_db`、`latent_shape`、`latent_std`。

## 实验日志

### 2026-06-18：frame-token 结构 smoke

- 形状验证：
  - representation: `(1,4,480,400)`
  - latent: `(1,32,50)`
  - pyramid: `[(1,32,400),(1,64,400),(1,128,200),(1,128,100),(1,128,50)]`
  - output representation: `(1,4,480,400)`
- 模型参数量约 `13.09M`。
- 随机 `training_step().backward()` finite。
- `encode/decode` smoke：`(1,2,96000) -> (1,32,50) -> (1,2,96000)`，输出 finite。

### 2026-06-18：固定评估脚本与 step0/quick20 基线

新增 `scripts/same_flow/eval_checkpoint.py`：

- 支持无 checkpoint 的 step0 随机模型评估。
- 支持 Lightning-style checkpoint 的 `raw` (`model.*`) 和 `ema` (`ema.ema_model.*`) 权重。
- 每条样本输出 `source.wav`、`reconstruct_{raw,ema}.wav` 和 `results.json`。

step0 随机模型命令：

```sh
python -m scripts.same_flow.eval_checkpoint \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_eval/step0_random \
  --max-items 2 \
  --denoising-steps 1 \
  --device cuda:0
```

结果：`corr_mean=0.00031`、`snr_db_mean=-0.0098`、`lsd_db_mean=49.06`、`latent_std_mean=0.0514`。这是随机初始化基线。

新增 `scripts/same_flow/debug_train.py`，用于非 Lightning 单卡快速训练和保存 checkpoint。quick20 命令：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/quick20 \
  --max-steps 20 \
  --batch-size 1 \
  --save-every 20 \
  --log-every 5 \
  --device cuda:0
```

quick20 训练能正常 backward/save，loss 在 `0.0067~0.1143` 间随 sigma 抖动，`latent_source_std=0.05769`。固定评估：

- raw: `corr_mean=0.00097`、`snr_db_mean=-0.0106`、`lsd_db_mean=49.61`。
- ema: `corr_mean=-0.00038`、`snr_db_mean=-0.0111`、`lsd_db_mean=49.61`。

结论：20 step 只证明训练/保存/评估链路通，尚未学到可辨认重建；需要跑至少 100/300/1000 step 再判断结构或采样是否有问题。

### 2026-06-18：debug loop scheduler 修复与 denoising anchor

问题：早期 `debug_train.py` 没有 step `configure_optimizers()` 返回的 LR scheduler。由于 `LinearLR(start_factor=1e-8)` 在构造时会把 optimizer lr 置为 `2e-12`，手写 loop 等于几乎没有更新参数；quick20/quick100 的 fixed eval 因此和 step0 基本一致。单步诊断显示有梯度 (`grad_norm=0.0937`) 但参数变化约 `7.7e-12`。

修复：`debug_train.py` 现在会提取并 step scheduler，step1 后 lr 到 `2e-6`，step100 到 `2e-4`。

同时为 `SameFlowTrainingWrapper` 新增可配置 `denoising_anchor_weight`，默认 `0.0`；`same-flow-overfit10.yaml` 当前设为 `1.0`。该项把 high-sigma prediction 直接拉向 clean representation，用于 from-scratch overfit 调试。纯 Music2Latent consistency 对照可把它改回 `0.0`。

quick100_anchor_sched 命令：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/quick100_anchor_sched \
  --max-steps 100 \
  --batch-size 1 \
  --save-every 100 \
  --log-every 20 \
  --device cuda:0
```

fixed eval raw/EMA 均加载完整 `1500` 个 key。2 条固定样本、1-step decode、`direct_sigma=10`：

- generative decode: `corr_mean=-0.00049`、`snr_db_mean=-0.0004`、`lsd_db_mean=40.65`。
- high-sigma direct denoise: `direct_corr_mean=0.0276`、`direct_snr_db_mean=0.0022`、`direct_lsd_db_mean=43.11`。

对比 step0 `lsd_db_mean≈49.06`，说明 scheduler 修复后模型开始改变生成输出，但 100 step 仍远未学会 waveform 相似性；下一步继续 300/1000 step 看 LSD 是否持续下降、corr/SNR 是否开始上升。

### 2026-06-18：quick300_anchor_sched

命令：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/quick300_anchor_sched \
  --max-steps 300 \
  --batch-size 1 \
  --save-every 300 \
  --log-every 50 \
  --device cuda:0
```

训练正常完成，step300 `loss=0.0570`、`lr=1.977e-4`、`latent_source_std=0.05769`。

1-step fixed eval (`direct_sigma=10`, 2 条固定样本)：

- raw/EMA 加载完整 `1500` 个 key，结果相同。
- generative decode: `corr_mean=0.00144`、`snr_db_mean≈0.0000`、`lsd_db_mean=38.83`。
- high-sigma direct denoise: `direct_corr_mean=0.0508`、`direct_snr_db_mean=0.0022`、`direct_lsd_db_mean=40.67`。

趋势：step0 LSD `49.06` -> quick100 LSD `40.65` -> quick300 LSD `38.83`，说明模型开始学会改善频谱距离；但 waveform corr/SNR 仍基本没起来。

4-step decode 对同一 checkpoint 明显更差：`corr_mean=-0.00214`、`snr_db_mean=-0.0471`、`lsd_db_mean=55.04`。当前随机 reverse diffusion 多步会继续注入噪声，短期调试先以 1-step decode 做主指标；后续应评估 deterministic sampling 或更接近训练目标的 sampler。

### 2026-06-18：CoDiCodec 理论对照与 sampler 诊断

用户提醒需要先理解一致性模型生成式重建原理。已读取 `docs/papers/codicodec.pdf` 的 3.1/3.2/4 节，关键点如下：

- 一致性 decoder 不是普通 autoencoder decoder，也不是预测噪声的 diffusion UNet；它学习 consistency function，把任意噪声轨迹点 `x_sigma` 直接映射到近 clean 的边界点 `x_sigma_min`。
- CoDiCodec/Music2Latent 路线使用 Consistency Training，不依赖预训练 diffusion teacher。训练时对同一 clean spectrogram 加相邻噪声层，student 处理高噪声点，stop-gradient teacher 处理低噪声点，用 Pseudo-Huber 距离约束两者输出一致。
- decoder 必须强依赖 encoder/upsampler cross-connections；论文明确说 consistency decoder 一步生成样本，因此 decoder 早期层必须拿到“要重建哪一个样本”的跨层信息。
- 推理从纯噪声 representation 开始，少步 denoise 到 clean。CoDiCodec parallel decoding 在后续 step 会把前一步 decoded chunks 加入逐步降低的噪声再 denoise；步数是质量/速度 trade-off。
- 论文训练细节：EDM parameterization、continuous log-normal noise sampling、指数 `Delta sigma` schedule，初始 `Delta t0=0.1`、最终 exponent `eK=2`。训练是长期大规模过程；CoDiCodec full model 约 150M 参数、单 A100 训练约两周。

对当前 same_flow 的结论：

- 当前 pair consistency 形式和论文/Music2Latent 的核心目标一致；`denoising_anchor_weight=1.0` 是为了 from-scratch overfit 加速的额外稳定项，不是论文原始单 loss。后续纯 consistency 对照应设回 `0.0`。
- 早期多步 stochastic decode 差，是因为每步会按 EDM schedule 重新注入很大噪声；在模型尚未学会高噪声 denoise 前，4-step stochastic 会被噪声拖坏。deterministic mode 更适合早期诊断。
- fixed eval 不能只看 waveform corr/SNR。当前模型先改善 LSD，说明频谱能量分布开始被学到；corr/SNR 不升说明相位/细节/时间对齐还没恢复，或者 latent cross-conditioning 还不够强。

新增 deterministic sampling：

- `Music2LatentSameFlow.decode(..., sampling_mode="stochastic"|"deterministic")`
- `SameFlowTrainingCallback` 支持 `sampling_mode`
- `eval_checkpoint.py` 支持 `--sampling-mode`
- `same-flow-overfit10.yaml` demo 默认改为 `sampling_mode: deterministic`

quick300 sampler sweep：

- 1-step deterministic: `lsd_db_mean=38.83`
- 2-step deterministic: `lsd_db_mean=39.78`
- 4-step deterministic: `lsd_db_mean=44.49`
- 8-step deterministic: `lsd_db_mean=45.63`
- 4-step stochastic: `lsd_db_mean=55.04`

结论：quick300 最佳仍是 1-step；stochastic 多步明显不适合早期模型。

### 2026-06-18：quick1000_anchor_sched

命令：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/quick1000_anchor_sched \
  --max-steps 1000 \
  --batch-size 1 \
  --save-every 500 \
  --log-every 100 \
  --device cuda:0
```

训练完成，step1000 `loss=0.03215`、`lr=1.561e-4`、`latent_source_std=0.05769`。

2 条样本评估：

- 1-step deterministic + `direct_sigma=10`: `lsd_db_mean=39.89`、`corr_mean=0.00180`、`direct_corr_mean=0.1994`。
- 2-step deterministic: `lsd_db_mean=41.63`、`corr_mean=0.00458`。
- 4-step deterministic: `lsd_db_mean=38.37`、`corr_mean=-0.00219`。

10 条 overfit 固定样本评估：

- 1-step deterministic + `direct_sigma=10`: `lsd_db_mean=31.88`、`corr_mean=0.00177`、`snr_db_mean=-0.0191`、`direct_corr_mean=0.1028`。
- 4-step deterministic: `lsd_db_mean=30.93`、`corr_mean=-0.00037`、`snr_db_mean=-0.9420`。

趋势：step1000 在 10 条上相对 step0/quick300 继续降低 LSD；direct high-sigma denoise 已经出现明显正相关，但 waveform corr/SNR 仍低。下一轮不要盲目只延长训练，应优先加诊断：

- 测 cross-conditioning 是否真的控制输出：同一个 initial noise 下交换 latent，看 prediction 是否跟随 latent/source。
- 记录 prediction/source RMS 和 representation-level MSE/LSD，区分“输出近静音/谱底下降”与“真实重建”。
- 尝试更接近论文 parallel decoding 的 refine sampler：先 denoise，再用较小 `sigma_cond` 加噪 refine，而不是直接 EDM 大跨度噪声或完全 deterministic。

下一步：

- 跑固定 overfit10 step0 评估，建立随机初始化基线。
- 起短训练，至少看 100/300/1000 step 的 fixed eval 指标和 demo。
- 若 `loss/total` 降而 fixed corr/SNR 不升，优先检查 reverse diffusion 采样与训练 consistency 目标是否匹配。

### 2026-06-18：CoDiCodec 理论驱动的 conditioning 链路修复

基于 CoDiCodec 论文的判断：一致性生成式重建能成立的前提不是“decoder 自己从噪声猜 clean”，而是 decoder 在每个噪声层都强条件于 encoder latent / upsampler cross-connections。若 latent 不携带输入，或 decoder pyramid 不把 latent 传到高分辨率层，训练会退化成近似无条件去噪，表现为输出 RMS 很低、LSD 可能下降但 corr/SNR 不起。

排查结果：

- 旧 `SameFlowResampling1d(type="encoder")` 在 stride-2 下采样时只取 learned output token。随机初始化下 adapter/input/stage0 仍能区分两条输入，但第一次 stride-2 transition 后差异塌到 `~1e-6`，最终不同真实音频 latent 完全相同。
- 只修 encoder 后，真实音频 latent 已不同，但 `SameFlowFrameDecoder` 的 upsample pyramid 除最深层外仍几乎相同；换 latent 后最终 denoised representation `out_diff≈2e-8`，说明 decoder upsampling 也在 learned output token 处丢条件。

修复：

- `scripts/same_flow/models.py` 的 `SameFlowResampling1d.forward()` 现在在没有 `override_new_tokens` 时，把每个 input chunk 的 mean content token 加到 learned output token 上，再送入 local transformer。encoder 和 decoder 都使用这一路径；仍保留 SAME 的 learned output token、chunk、sliding-window transformer 和 mapping 结构。
- 这个改动没有引入外部组件，也没有改变 skip 连接、loss 或配置接口。
- `scripts/same_flow/eval_checkpoint.py` 新增 `mismatch_direct_delta_rms`，用于直接量化同一 noisy representation 换 latent 后输出变化，避免只看 corr/LSD 掩盖 conditioning 强度。

验证：

- 随机模型形状诊断：`rep=(2,4,480,400)`，adapter diff `0.673`，`trans1` diff `0.0976`、`trans2` diff `0.0710`、`trans3` diff `0.0507`，最终 latent diff `0.1785`、latent std `0.2956`；不再在 stride-2 下采样后塌缩。
- 真实 dataloader batch：`rep=(2,4,480,400)`，latent `(2,32,50)`，两条音频 latent distance `0.156`。
- 修 decoder seeded token 后，用旧 quick100 checkpoint 直接诊断：decoder pyramid mismatch diff 从高到低约 `0.034/0.059/0.104/0.100/0.096`，最终 `out_diff=0.00405`，不再是 `2e-8`。
- 两个补丁后重新跑 `quick100_content_seeded_v2`：step100 `loss=0.109953`、`lr=2e-4`、`latent_std=0.44802`。4 条 fixed eval：`lsd_db_mean=35.91`、`corr_mean=0.00096`、`direct_corr_mean=0.02436`、`prediction_rms_mean=0.00174`、`direct_rms_mean=0.00232`。自定义 mismatch 诊断：latent pair distance 约 `0.048~0.052`，matched representation RMS `0.0458`，mismatch delta RMS `0.000531`，相对约 `1.16%`。正式 2 条 smoke 的 `mismatch_direct_delta_rms_mean=2.98e-05`。

结论：

- 结构级 collapse 已修复：encoder latent 不再常数，decoder pyramid/输出开始受 latent 影响。
- 训练级问题仍存在：100 step 下 conditioning 影响偏弱，输出 RMS 仍远低于 source RMS，不能把当前 LSD 下降解释成真实重建成功。
- 下一步应从两个补丁后的代码跑更长 overfit（建议 300/1000 step），主看 `prediction_rms`、`direct_rms`、`mismatch_direct_delta_rms` 是否随训练上升，以及 corr/SNR 是否跟随；若 delta 仍长期只有 1% 左右，应增强 latent 注入（例如在 denoiser 每层加入 AdaLN/FiLM 或 residual prior），而不是继续只调 sampler。

### 2026-06-18：content-seeded 主线继续训练与 residual prior 对照

从 content-seeded resampling 后的无 prior 主线继续做短 overfit。命令示例：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/quick2000_content_seeded_v2 \
  --max-steps 2000 \
  --batch-size 1 \
  --save-every 1000 \
  --log-every 200 \
  --device cuda:0
```

无 prior 主线结果：

- quick300 (`tmp/same_flow_debug_train/quick300_content_seeded_v2/last.ckpt`)，4 条 eval：`corr_mean≈0.000006`、`lsd_db_mean=37.06`、`prediction_rms_mean=9.73e-5`、`direct_corr_mean=0.0433`、`mismatch_direct_delta_rms_mean=1.70e-5`。300 step 仍很弱。
- quick1000 (`tmp/same_flow_debug_train/quick1000_content_seeded_v2/last.ckpt`)，10 条 1-step deterministic eval：`corr_mean=0.0685`、`snr_db_mean=-0.0084`、`lsd_db_mean=31.02`、`prediction_rms_mean=0.000777`、`direct_corr_mean=0.1305`、`direct_rms_mean=0.00240`、`mismatch_direct_delta_rms_mean=0.00133`。
- quick2000 (`tmp/same_flow_debug_train/quick2000_content_seeded_v2/last.ckpt`)，10 条 1-step deterministic eval：`corr_mean=0.0964`、`snr_db_mean=0.0172`、`lsd_db_mean=32.29`、`prediction_rms_mean=0.00129`、`direct_corr_mean=0.1560`、`direct_rms_mean=0.00299`、`mismatch_direct_delta_rms_mean=0.00358`。

结论：无 prior 主线从 1000 到 2000 step 继续提升 waveform corr、prediction RMS 和 mismatch delta，说明 conditioning 链路会随训练增强；但输出 RMS 仍远低于 source RMS (`0.1086`)，还没有达到可用重建。

采样步数对 quick2000：

- 1-step deterministic：`corr_mean=0.0964`、`snr_db_mean=0.0172`、`prediction_rms_mean=0.00129`。
- 2-step deterministic：`corr_mean=0.1025`、`snr_db_mean=-1.648`、`prediction_rms_mean=0.01483`。
- 4-step deterministic：`corr_mean=0.0952`、`snr_db_mean=-2.076`、`prediction_rms_mean=0.01820`。

2/4-step 会把能量拉大，但误差也明显放大；当前主诊断仍用 1-step deterministic，2-step 只能作为听感候选，不应用作主训练判断。

尝试 residual prior 分支：

- 代码中保留可选 `model.use_latent_prior`：latent decoder 先预测 clean representation prior，denoiser 在 `noisy - prior` 残差空间做 consistency，输出 `prior + residual`。
- wrapper 保留 `latent_prior_loss_weight`，先试 pseudo-Huber prior loss，再改 MSE prior loss。
- pseudo-Huber prior (`latent_prior_loss_weight=1.0`) quick300：10 条 `corr_mean=0.00091`、`prediction_rms_mean=0.000138`、`direct_corr_mean=0.0315`、`mismatch_direct_delta_rms_mean=3.79e-5`。prior 自身 `prior_rms≈0.0063`、`prior_corr≈0.0011`，基本塌零。
- MSE prior (`latent_prior_loss_weight=5.0`) quick300：10 条 `corr_mean=0.0080`、`prediction_rms_mean=0.000174`、`direct_corr_mean=0.0363`、`mismatch_direct_delta_rms_mean=7.98e-5`。prior 自身仍 `prior_rms≈0.0050`、`prior_corr≈0.0014`，没有学成粗重建。

决策：

- 默认 `same-flow-overfit10.yaml` 已回到 `use_latent_prior=false`、`latent_prior_loss_weight=0.0`，因为当前 residual prior 分支短训明显弱于无 prior 主线。
- 保留 residual prior 代码作为实验开关，但不是当前推荐路径。
- 下一步优先继续拉长无 prior content-seeded 主线到 3000/5000 step，或增强 denoiser 层内 latent 注入（例如 AdaLN/FiLM），不要继续用当前 prior head 配置消耗训练时间。

### 2026-06-19：denoiser 层内 latent FiLM

动机：content-seeded resampling 修复了 encoder/decoder token collapse，继续训练也能提升 conditioning，但无 FiLM 主线到 2000 step 仍只有 `corr_mean=0.0964`、`prediction_rms=0.00129`。这说明 latent 已经能影响输出，但强度还不够。当前改动在每个 denoise block 前增加 token-wise latent FiLM：同尺度 latent feature `d` 不只与 `x` 相加，还通过一个轻量 `Linear(C,2C)` 生成 per-token scale/shift 调制 `x`。FiLM projection 零初始化，初始行为接近旧模型，训练中逐渐学习层内条件注入。

代码/配置：

- `scripts/same_flow/models.py` 新增 `LatentFiLM1d`。
- `Music2LatentSameFlow` 新增 `use_latent_film`，在 down/up denoise blocks 中分别用 `down_latent_films` / `up_latent_films` 调制。
- `scripts/configs/experiment/same-flow-overfit10.yaml` 当前 `use_latent_film: true`，`use_latent_prior: false`。
- smoke：`py_compile` 通过；随机 forward `z=(2,32,50)`、`out=(2,4,480,400)` finite；真实 batch `training_step().backward()` finite。

短训结果：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/quick2000_latent_film \
  --max-steps 2000 \
  --batch-size 1 \
  --save-every 1000 \
  --log-every 200 \
  --device cuda:0
```

FiLM quick300 仍弱：`corr_mean=0.00147`、`prediction_rms=0.000102`、`direct_corr=0.0320`、`mismatch_direct_delta_rms=1.78e-5`。

FiLM quick1000：`corr_mean=0.0929`、`snr_db_mean=0.0117`、`prediction_rms=0.000645`、`direct_corr=0.1510`、`mismatch_direct_delta_rms=0.00134`。corr 明显高于无 FiLM quick1000 (`0.0685`)，但 RMS 仍低。

FiLM quick2000：`corr_mean=0.1250`、`snr_db_mean=0.0447`、`lsd_db_mean=31.97`、`prediction_rms=0.00360`、`direct_corr=0.1828`、`direct_rms=0.00529`、`mismatch_direct_delta_rms=0.00725`。对比无 FiLM quick2000：`corr_mean=0.0964`、`prediction_rms=0.00129`、`direct_corr=0.1560`、`mismatch_direct_delta_rms=0.00358`。结论：FiLM 到 2000 step 后开始明显增强 latent conditioning。

长训结果：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/quick5000_latent_film_sched6000 \
  --max-steps 5000 \
  --batch-size 1 \
  --save-every 2500 \
  --log-every 500 \
  --device cuda:0 \
  --override wrapper.lr_schedule_total_steps=6000 \
  --override wrapper.consistency_step_total_steps=6000
```

最终 checkpoint：`tmp/same_flow_debug_train/quick5000_latent_film_sched6000/last.ckpt`。

10 条 fixed eval：

- 1-step deterministic：`corr_mean=0.2762`、`snr_db_mean=0.4373`、`lsd_db_mean=32.04`、`prediction_rms=0.01731`、`direct_corr=0.3120`、`direct_rms=0.02107`、`mismatch_direct_corr=0.0491`、`mismatch_direct_delta_rms=0.03273`。
- 2-step deterministic：`corr_mean=0.2500`、`snr_db_mean=-1.2686`、`prediction_rms=0.03966`、`mismatch_direct_delta_rms=0.03280`。
- 4-step deterministic：`corr_mean=0.2129`、`snr_db_mean=-2.2883`、`prediction_rms=0.04197`、`mismatch_direct_delta_rms=0.03200`。

逐样本看，部分样本已明显过拟合：item 1 `corr=0.7662`、item 7 `corr=0.6337`；但 item 8/9 仍几乎没学到。这表示模型已能用 latent 做生成式重建，但还不稳定，10 条 overfit 尚未全覆盖。

结论：

- `use_latent_film=true` 是当前推荐主线，明显强于无 FiLM 和 residual prior。
- 当前最佳主评估仍是 1-step deterministic。2/4-step 会增加能量，但 SNR 明显变差。
- 输出 RMS 已从无 FiLM 2000 的 `0.00129` 提升到 FiLM 5000 的 `0.01731`，但仍低于 source RMS `0.1086`，所以目标还没完成。
- 下一步应继续 FiLM 主线到更长 step，或提高模型/数据表达能力；同时应听 `tmp/same_flow_eval/quick5000_latent_film_sched6000_raw_10_step1/` 下的 wav，确认高 corr 样本是否可听重建，低 corr 样本是否是静音/难样本或模型局部失败。

### 2026-06-19：可听推理样例与幅度诊断

针对用户要求“推理一下音频我听一下”，已把当前最佳 FiLM 5000 checkpoint 的代表样本整理到 `tmp/same_flow_listen/`。每个文件都是 `source -> 0.35s silence -> reconstruct`：

- `tmp/same_flow_listen/best_item001_corr0.766_source_then_reconstruct.wav`
- `tmp/same_flow_listen/best_item007_corr0.634_source_then_reconstruct.wav`
- `tmp/same_flow_listen/mid_item004_corr0.373_source_then_reconstruct.wav`
- `tmp/same_flow_listen/fail_item008_corr0.002_source_then_reconstruct.wav`
- 拼接总览：`tmp/same_flow_listen/same_flow_best_mid_fail_source_then_reconstruct.wav`

当前最佳评估目录仍是 `tmp/same_flow_eval/quick5000_latent_film_sched6000_raw_10_step1/`，checkpoint 是 `tmp/same_flow_debug_train/quick5000_latent_film_sched6000/last.ckpt`。

逐样本现状：

- 成功/明显拟合案例：item 1 `corr=0.7662`，item 7 `corr=0.6337`。
- 中等案例：item 4 `corr=0.3732`，item 5 `corr=0.2939`。
- 失败案例：item 8 `corr=0.0019`，item 9 `corr=0.0434`；item 6 的 source RMS 只有 `0.00211`，接近静音，不能用 corr 单独判断。

`eval_checkpoint.py` 已新增 optimal-gain 指标，用于判断是否主要是幅度偏低。对当前最佳 checkpoint 重评到 `tmp/same_flow_eval/quick5000_latent_film_sched6000_raw_10_step1_gain/`：

- 原始 1-step：`corr_mean=0.2782`、`snr_db_mean=0.4394`、`prediction_rms_mean=0.01756`。
- 最优线性增益：`optimal_gain_mean=2.31`、`optimal_gain_snr_db_mean=0.7622`、`optimal_gain_prediction_rms_mean=0.04106`。

结论：确实有明显幅度偏低问题，但 optimal gain 只把 SNR 从约 `0.44 dB` 提到 `0.76 dB`，说明不是纯音量问题，内容/相位细节仍不足。

`debug_train.py` 已支持 `--checkpoint` / `--checkpoint-variant` 续训。当前后台有一个从 FiLM 5000 checkpoint 继续的低 LR run：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-overfit10 \
  --output-dir tmp/same_flow_debug_train/finetune5000_latent_film_from5000_lr5e5 \
  --max-steps 5000 \
  --batch-size 1 \
  --save-every 2500 \
  --log-every 500 \
  --device cuda:0 \
  --checkpoint tmp/same_flow_debug_train/quick5000_latent_film_sched6000/last.ckpt \
  --override wrapper.lr_schedule=constant \
  --override wrapper.learning_rate=5e-5
```

启动验证：checkpoint 加载 `loaded=1549, missing=0, skipped=0`，step1 finite。该 run 在用户查询时仍在运行，尚未评估结果。

### 2026-06-19：改为逐步验证的 max-denoise 最小实验

用户指出当前 1-step 推理评估下，应先把问题简化为“固定最大噪声到干净频谱”的监督目标，而不是直接等待完整 consistency schedule。已建立独立目录 `scripts/same_flow_debug/`，从 `scripts/same_flow/` 复制而来，作为结构/训练目标调试区，不破坏现有基线。

新增内容：

- `scripts/same_flow_debug.models.Music2LatentSameFlow(use_patch_denoiser=true)`：尝试 patch-grid denoiser，把 `[B,4,480,T]` 按 32-bin 分成 15 个 patch，每个 patch 作为独立时间序列跑 SAME local transformer，latent/pyramid 条件按 patch repeat 注入。
- `scripts/configs/experiment/same-flow-debug-overfit10.yaml`：debug 配置，默认指向 `scripts.same_flow_debug.*`。
- `scripts/same_flow_debug.wrapper.SameFlowTrainingWrapper(training_objective="max_denoise")`：固定 `sigma=max_denoise_sigma or sigma_max`，直接训练 `model(clean, clean + sigma * noise) -> clean representation`。
- `clean_loss_type=mse|pseudo_huber`：用于对比 pseudo-Huber 和 MSE。
- `debug_train.py --probe-every N --fixed-probe-batch`：固定同一条样本、同一 latent、同一噪声、同一 `sigma=80`，每 N step 输出 `probe_repr_mse`、`probe_snr_db`、`probe_lsd_db`、`probe_hf_ratio`、`probe_pred_rms`。

预期行为：

- 在固定噪声 max-denoise 单样本 overfit 下，若模型路径合理，`probe_repr_mse` 应持续下降，`probe_snr_db` 应持续上升，`probe_pred_rms` 应从低值接近 source RMS，`probe_hf_ratio` 应逐步向 1 靠近。
- 若 loss 下降但 `probe_pred_rms` 下降到近 0，说明模型在利用损失/参数化退化到低能量输出。
- 若 SNR/RMS 上升但 `probe_hf_ratio` 长期接近 0，说明模型先学低频，高频学习慢，需要更长训练或高频加权/结构约束。

关键结果：

1. patch-grid denoiser 第一版不合格。`maxdenoise_probe20_patch` 中 `loss` 和 `LSD` 下降，但 `probe_pred_rms` 从 `0.0061` 降到 `0.0035`，趋向低能量输出。step500 的 patch-grid 多组实验也几乎近静音，暂不作为主线。
2. frame-token 原结构 + FiLM 5000 checkpoint + 固定噪声 + pseudo-Huber 能学习：`fixed_noise_frame_preload_100` 在 100 step 内 `probe_snr_db 0.16 -> 1.08`、`probe_lsd_db 35.66 -> 30.80`、`probe_pred_rms 0.0186 -> 0.0525`，但 `probe_hf_ratio` 仍约 `0.024`。
3. 固定噪声 + MSE 比 pseudo-Huber 更适合最小 overfit。`fixed_loss_g5_mse_lr1e3` 100 step 得到 `probe_snr_db=2.99`、`probe_pred_rms=0.0828`、`probe_hf_ratio=0.117`。
4. 继续同一固定噪声 MSE run 到 500 step 后明显过拟合成功：
   - checkpoint：`tmp/same_flow_debug_train/fixed_loss_g5_mse_lr1e3_cont500/last.ckpt`
   - `probe_mse=0.00663`
   - `probe_snr_db=17.26`
   - `probe_lsd_db=26.11`
   - `probe_hf_ratio=0.687`
   - `probe_pred_rms=0.1943`
   - source RMS 导出检查为 `0.2020`，prediction RMS 为 `0.1940`

可听文件：

- `tmp/same_flow_debug_listen/fixed_noise_maxdenoise_step500_source_then_reconstruct.wav`
- `tmp/same_flow_debug_listen/fixed_noise_maxdenoise_step500_reconstruct.wav`

结论：

- local transformer 不是“没理由但实际不行”；在最小固定样本、固定噪声、固定 `sigma=80` 的单步生成式去噪任务下，它可以明确学习并逐步恢复高频。
- 之前的失败主要来自目标过复杂、sigma 分布/consistency pair 对 from-scratch overfit 不友好，以及 pseudo-Huber 在该阶段容易让低频/低能量路径先占优。
- 下一步应从这个最小可行点逐步放宽：固定噪声多样本 -> 随机噪声单样本 -> 随机噪声多样本 -> 固定 high sigma 多样本 -> 再回到 consistency pair/sigma schedule。每一步都用 `probe_*` 指标观察，不再只等完整 demo。

### 2026-06-19：baseline 约束修正与随机噪声阶段结果

用户明确要求：`music2latent-training (1)/music2latent-training` baseline 不能改训练行为参数；可以缩小模型参数量，但不能改会决定行为的 loss、sigma schedule、STFT 表示、训练公式、skip/conditioning 逻辑。此前临时写的 `scripts/same_flow_debug/original_music2latent_baseline.py` 直接复刻部分训练 loop 并改了 hparams，只能算错误方向 smoke，不作为 baseline 结论。

SAME-flow debug 已确认的阶段性结果：

- 固定 source + 固定噪声 + `sigma=80` + max-denoise MSE：`tmp/same_flow_debug_train/fixed_loss_g5_mse_lr1e3_cont500/last.ckpt`，step500 `probe_mse=0.00663`、`probe_snr=17.26 dB`、`probe_lsd=26.11`、`probe_hf_ratio=0.687`、`probe_rms=0.1943`。这是最小可行 baseline。
- 固定 source + 随机噪声 + `sigma=80` 从固定噪声成功点续训，8 卡对照 step500：
  - `randnoise_g7_from_fixed_hf1_start200`：`mse=0.00543`、`snr=19.16`、`lsd=23.75`、`hf=0.681`、`rms=0.1972`。
  - `randnoise_g2_from_fixed_lr1e3`：`mse=0.00540`、`snr=19.05`、`lsd=24.93`、`hf=0.766`、`rms=0.1962`。
  - `randnoise_g3_from_fixed_hf10`：`mse=0.00560`、`snr=18.85`、`lsd=22.63`、`hf=0.732`、`rms=0.1951`。
  结论：单样本随机噪声可学，不只是记住固定噪声；高频 anchor 改善 LSD，但不是最终训练目标。
- 10 样本 max-denoise 多样本对照失败：`multi10_*` 1000 step 多数 probe SNR 约 `0 dB`，最稳的 `multi10_g2_lr1e4` step1000 也只有 `probe_snr=1.75`、`probe_lsd=29.12`、`probe_hf=0.078`、`probe_rms=0.1111`。这说明多样本阶段不能只看 representation MSE 的 loss，需要和 Music2Latent 对照逐层行为，不能额外加非官方 loss。

官方 Music2Latent baseline 设置：

- 10 条 2 秒缓存音频已导出为 wav：`tmp/original_music2latent_official_overfit10_wav/`，共 10 个 wav、约 7.6MB，避免 raw mp3 解码成为瓶颈。
- 官方训练入口仍使用 `tmp/music2latent-training (1)/music2latent-training/launch.py` / `music2latent.train.Trainer` / `music2latent.models.UNet` / 官方 `train_it()` 一致性训练方式。
- 当前启动配置 `tmp/original_music2latent_official_overfit10_small_config.py` 只改：数据路径、输出路径、`compile_model=false`、`num_workers=0`、FAD worker/sample 数、以及容量参数 `base_channels=32,bottleneck_base_channels=128,cond_channels=128,heads=2,bottleneck_channels=32`。不改 `lr`、`total_iters`、`warmup_steps`、`data_length`、`hop`、`alpha/beta_rescale`、`sigma_*`、`schedule`、`use_lognormal`、loss 或训练公式。
- 由于环境无法加载 `laion_clap` 的 `bert-base-uncased` tokenizer，训练通过 `tmp/original_music2latent_official_train_shim.py` 注入最小 `laion_clap` stub 以绕过 import 阶段 FAD 依赖。该 shim 只允许训练 early steps；如果到 epoch 末需要 FAD，会报错，不用于 FAD 指标。
- 当前后台 run 日志：`logs/original_music2latent_official/official_small_overfit10.log`；进程在 GPU4；启动后扫描到 10 个 train/test wav，参数量 `11220130`。

下一步：不要再引入额外 loss。应做同一 batch 上的逐层行为对照：官方卷积 Music2Latent 与 SAME-flow 的 encoder latent std/shape、decoder pyramid RMS、denoiser down/up 注入项 RMS、skip RMS、输出 RMS，以及 mismatch latent 对输出影响。

补充：临时 `scripts/same_flow_debug/original_music2latent_baseline.py` 已删除，避免误用。有效 Music2Latent baseline 只保留官方代码路径：`tmp/original_music2latent_official_train_shim.py` + `tmp/original_music2latent_official_overfit10_small_config.py`。shim 只绕过 FAD import，训练 step 仍进入官方 `Trainer.train_it()`。

### 2026-06-19：官方 Music2Latent baseline 必须先证明可重建

用户纠正：不能只启动 `music2latent-training` 就把它当 baseline，必须先验证它确实能在小数据 overfit 上重建音频。已新增短程验证 harness：

- `tmp/original_music2latent_official_short_eval.py`：加载官方 repo、实例化官方 `music2latent.train.Trainer`，训练步调用官方 `Trainer.train_it()`；评估时用官方 `to_representation_encoder()`、`model.encoder()` 和 `music2latent.utils.generate(latents=...)` 导出 `source.wav`、`reconstruct_raw.wav`、`reconstruct_ema.wav`，并记录 `corr/SNR/LSD/RMS/latent_shape`。
- 因当前环境 `laion_clap` import 会在 BERT tokenizer 上失败，脚本只 stub 掉 import-time FAD 依赖；训练公式、模型 forward、sigma schedule 和 reverse diffusion 不通过 stub。
- `tmp/original_music2latent_official_overfit10_wav/` 是从 same_flow overfit10 缓存导出的 10 条 2 秒 wav，用于避免 mp3 解码成为瓶颈。

先跑的 `official-small` strict 调度配置 `tmp/original_music2latent_official_overfit10_small_config.py` 只改路径、runtime 和容量，不改官方 `lr/warmup/total_iters`。结果不能证明重建：

- 1000 step 仍在官方 `warmup_steps=10000` 冷启动内，LR 约 `1e-5`。
- step1000 EMA：`corr_mean=-0.0004`、`snr_db_mean≈0`、`lsd_db_mean=37.53`、`prediction_rms_mean=8.6e-5`、`source_rms_mean=0.1516`，几乎静音。
- 同 checkpoint raw weights 也几乎静音：`prediction_rms_mean=1.09e-4`、`corr_mean=0.0003`。

因此 strict-small 目前只说明官方路径能跑，不能作为“已验证可重建 baseline”。原因主要是官方大规模训练调度不适合 1000-step 快速验证。

已改用独立 fast-overfit 配置 `tmp/original_music2latent_official_overfit10_fast_config.py`，允许只改快速验证参数：

- 可改：`batch_size=4`、`lr=3e-4`、`total_iters=3000`、`iters_per_epoch=3000`、`warmup_steps=100`、`ema_momentum=0.99`、容量参数。
- 不改：官方 STFT 表示 (`hop=512, alpha=0.65, beta=0.34`)、mono real/imag `data_channels=2`、sigma 分布和 schedule、consistency pair 训练公式、`inference_diffusion_steps=1`、官方随机 reverse diffusion。
- 当前 fast-overfit 容量：`base_channels=32,bottleneck_base_channels=256,cond_channels=128,heads=4,num_bottleneck_layers=2,bottleneck_channels=64`，参数量约 `13.1M`。

fast-overfit 运行日志：`logs/original_music2latent_official/official_short_eval_fast.log`，输出目录：`tmp/original_music2latent_official_short_eval_fast/`。step500 已不再静音，但还没有重建相关性：

- raw：`corr_mean=-0.0016`、`snr_db_mean=-0.252`、`lsd_db_mean=43.40`、`prediction_rms_mean=0.0359`。
- EMA：`corr_mean=-0.0071`、`snr_db_mean=-0.168`、`lsd_db_mean=41.37`、`prediction_rms_mean=0.0283`。

下一步继续看 step1000/1500。如果 fast-overfit 后仍不能重建，不能直接说 SAME-flow 差，应先确认官方 Music2Latent 在同数据、同验证脚本下是否真的能重建；必要时提高容量或训练到 3000/5000 step，但仍不改官方目标函数。

用户试听后指出 official 重建“其实还行，只是有许多底噪”，因此原先只看时域 `corr/SNR` 的评价口径过严/不完整。已新增通用指标脚本：

```sh
python tmp/reconstruction_audio_metrics.py \
  --pair SOURCE.wav RECONSTRUCT.wav \
  --output tmp/reconstruction_metric_reports/report.json
```

新主指标：

- `mrstft_sc`：multi-resolution STFT spectral convergence，越低越好。
- `mrstft_log_mag_l1`：多尺度 log magnitude L1，越低越好。
- `logmel_l1/logmel_l2`：log-mel 距离，越低越好。
- `low_energy_ratio/high_energy_ratio`：预测相对 source 的低频/高频能量比，用于判断低频偏置、底噪和高频缺失。
- `corr/SNR` 继续保留，但只作辅助；生成式一致性重建中，采样底噪和相位扰动会显著拉低它们。

代表结果写入 `tmp/reconstruction_metric_reports/official_and_same_selected.json`：

| 样例 | corr | SNR | MRSTFT-SC | MRSTFT log-mag | logmel L1 | low ratio | high ratio | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| official overfit1 step3000 raw `generate` | 0.101 | -2.37 | 0.619 | 3.185 | 1.832 | 0.924 | 2.325 | 听感有内容但底噪/高频噪声重，不能用 corr 单独否定 |
| official overfit1 step3000 EMA `generate` | 0.050 | -2.50 | 0.600 | 3.241 | 1.848 | 0.874 | 2.088 | EMA 与 raw 接近，仍有底噪 |
| official overfit1 step3000 raw `direct_sigma80` | 0.154 | -2.01 | 0.599 | 3.175 | 1.824 | 0.882 | 2.185 | 与 generate 接近，说明 `sigma=80` 本身较难 |
| official overfit1 step3000 raw `direct_sigma10` | 0.808 | 4.24 | 0.375 | 2.448 | 1.389 | 0.960 | 1.223 | 中噪声 direct denoise 已明显重建 |   
| official overfit1 step3000 raw `direct_sigma1` | 0.986 | 15.59 | 0.129 | 2.755 | 1.477 | 0.967 | 1.828 | 低噪声 direct denoise 基本对齐时域，但仍有高频能量偏差 |
| SAME debug fixed source/noise `sigma80` MSE step500 | 0.990 | 16.74 | 0.125 | 1.906 | 1.216 | 0.924 | 0.460 | 最小固定噪声任务明显过拟合成功，但高频偏少 |

这个对照改变后续判断：

- official Music2Latent 的“可重建”不能只按 `corr≈0.1` 判失败，听感和 STFT/mel 指标都说明其生成结果已保留部分内容，但底噪明显。
- SAME-flow 固定噪声最小任务已经优于 official `sigma80 generate/direct` 的 STFT/mel 指标，但它是更简单的固定噪声监督，不等价于完整 consistency baseline。
- 下一步 SAME-flow 应按难度递增：固定/随机噪声 `sigma=1/10/80` direct denoise 指标曲线 -> 多样本 direct denoise -> 再回完整 consistency `generate`。不要跳过中噪声阶段直接用 `sigma_max=80` 一步生成判定结构失败。

已给 `scripts/same_flow_debug/debug_train.py` 增加 `--probe-sigma`，并在 probe 日志里加入 `probe_mrstft_sc`、`probe_mrstft_log_mag_l1`、`probe_logmel_l1/l2`；`scripts/same_flow_debug/eval_checkpoint.py` 同步加入 MR-STFT/log-mel 汇总。默认行为不变。

固定单样本 + 随机噪声 + MSE direct denoise 的 300 step sigma sweep：

| SAME debug run | sigma | probe SNR | probe MRSTFT-SC | probe logmel L1 | probe HF ratio | probe RMS | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tmp/same_flow_debug_train/sigma_sweep_fixed1_rand_sigma1_step300` | 1 | 8.08 | 0.345 | 2.628 | 4.550 | 0.1689 | 低噪声可快速学到内容，但高频/底噪过量 |
| `tmp/same_flow_debug_train/sigma_sweep_fixed1_rand_sigma10_step300` | 10 | 2.25 | 0.715 | 2.550 | 0.286 | 0.0912 | 中噪声开始明显困难，高频不足 |
| `tmp/same_flow_debug_train/sigma_sweep_fixed1_rand_sigma80_step300` | 80 | 0.00 | 0.999 | 4.430 | 0.287 | 0.0010 | 高噪声 300 step 基本近静音，需更长训练或从成功 checkpoint 续训 |

和 official probe 对照：

- official overfit1 step3000 `direct_sigma10`：`corr=0.808`、`SNR=4.24`、`MRSTFT-SC=0.375`、`logmel_l1=1.389`、`high_energy_ratio=1.223`。
- SAME `sigma=10` 300 step 距离 official direct_sigma10 还有明显差距，尤其 MRSTFT-SC 和高频能量。
- SAME `sigma=1` 300 step 在 MRSTFT-SC 上接近 official `direct_sigma10`，但高频比例过高，说明低噪声路径已通，中高噪声和频谱平衡仍是问题。

下一步实验决策：

1. `sigma=10` 从 300 step checkpoint 继续到 1000，看 MRSTFT/logmel 是否能追近 official `direct_sigma10`。
2. `sigma=80` 不应只看 300 step；从已成功的 `fixed_loss_g5_mse_lr1e3_cont500` 或随机噪声成功 checkpoint 继续，观察是否稳定保持高频并降低底噪。
3. 若 `sigma=10` 长训仍高频不足，再检查 SAME denoiser 的频率 patch 表示和 FiLM/skip 注入强度，而不是先回完整 consistency。

### 2026-06-19：训练方式和 loss 对齐到 Music2Latent

用户要求：后续主要排查模型问题，因此用于 loss 对比的训练方式和 loss 必须和 `music2latent-training` 完全统一。

已核对官方 Music2Latent：

- `train_it()`：`data=to_representation(wv)`、`data_encoder=to_representation_encoder(wv)`。
- sigma 采样：`use_lognormal=True` 时先构造 `10000` 个离散 sigma 权重 `gaussian_pdf(log(sigma))`，再 `torch.multinomial()`，最后加 uniform jitter 除以 `9999`。
- step schedule：`base_step=0.1,start_exp=1,end_exp=3,total_iters` 指数调度，step 从 `0.1` 逐步到 `0.001`。
- noisy pair：同一个 `noise` 分别加到 `sigma_step` 和 `sigma` 上。
- forward：`fdata = model(... sigma_step).detach()`，`fdata_plus_one = model(... sigma)`。
- loss：`huber(fdata, fdata_plus_one, get_loss_weight(sigmas, sigmas_step))`，其中 huber 是 per-sample flatten 后先求平方和，再 pseudo-Huber，再乘 `1/(sigma_high-sigma_low)`。
- optimizer：`RAdam(lr=1e-4, betas=(0.9,0.999))`，官方 cosine/warmup schedule，EMA 默认开启。

已完成代码对齐：

- `scripts/same_flow.wrapper` 和 `scripts/same_flow_debug.wrapper`：
  - 新增 `official_huber_loss()`，与官方 `music2latent.utils.huber()` 数值一致；随机 tensor smoke：`abs_diff=0.0`。
  - `sigma_sampling=lognormal` 改为官方 `10000`-bin weighted multinomial + jitter；旧直接 `exp(N)` 采样只保留为 `direct_lognormal` 调试选项。
  - consistency pair loss 改为 official huber + `w=1/(sigma_high-sigma_low)`，不再使用逐元素 `pseudo_huber_loss(...).mean()`。
  - `denoising_anchor_weight` 和 `high_freq_anchor_weight` 默认保持 `0.0`；打开它们会改变 official loss，只能用于 debug，不能用于 Music2Latent loss 对比。
  - 新增 `lr_schedule=music2latent_cosine`，复刻官方 warmup + cosine 到 `final_learning_rate` 的调度。
- `scripts/configs/experiment/same-flow-overfit10.yaml` 和 `same-flow-debug-overfit10.yaml`：
  - `optimizer_name: radam`
  - `lr_schedule: music2latent_cosine`
  - `final_learning_rate: 1e-6`
  - `consistency_step_end_exp: 3.0`
  - `sigma_sampling: lognormal`
  - `denoising_anchor_weight: 0.0`
  - `high_freq_anchor_weight: 0.0`
  - debug 配置显式 `training_objective: consistency`

仍然不完全相同、但属于有意的模型/表示差异：

- official Music2Latent 是 mono waveform -> real/imag STFT `data_channels=2`、`sample_rate=44100`、`hop=512`、latent `64`。
- SAME-flow 目标是 stereo 48 kHz，当前 STFT 表示为 `[B,4,480,T]`、`hop=240`，latent 目标 `32`，主干是 SAME local transformer。这些是要排查的模型/表示差异，不是训练 loss 差异。
- `max_denoise`、`clean_loss_type=mse`、`--fixed-source-random-noise` 和 `--probe-sigma` 都是模型能力探针，不能用于和 Music2Latent 的 loss 数值直接对比。

刚完成的非 official-loss 探针结果也记录清楚，避免误用：

- `tmp/same_flow_debug_train/sigma_sweep_fixed1_rand_sigma10_from300_to1000/last.ckpt` 是 `max_denoise + MSE` 续训结果，不用于 loss 对比；但它说明 SAME 在 `sigma=10` direct denoise 上可学，step1000 probe：`probe_mrstft_sc=0.1844`、`probe_logmel_l1=1.8649`、`probe_hf_ratio=0.2497`、`probe_snr_db=13.47`、`probe_pred_rms=0.1875`。
- `tmp/same_flow_debug_train/sigma80_from_randnoise_g2_cont500/last.ckpt` 也是非 official-loss 探针；从先前成功点续训后退化，step500 probe：`probe_mrstft_sc=0.9223`、`probe_logmel_l1=2.6207`、`probe_hf_ratio=0.2568`、`probe_snr_db=0.43`、`probe_pred_rms=0.0264`。后续不再用它比较 Music2Latent loss。

下一步应基于已对齐的 official consistency 训练口径，启动小规模 SAME-flow consistency run，并只把它的 `loss/consistency_pair` 与 official baseline loss 趋势对比；模型排查重点看 latent/std、pyramid/skip 能量、denoiser output RMS、direct sigma=1/10/80 的 STFT/mel 指标。

loss 对比补充：

- official huber 公式 smoke 已与 SAME wrapper 数值严格一致：同一随机 `x/y/w` 上 `abs_diff=0.0`。
- 已跑 official fast 20-step loss probe：`tmp/original_music2latent_official_loss_probe20/train_log.json`。step1/5/10/15/20 loss 分别约 `98.61/141.02/123.53/188.04/209.30`，20-step recent mean 约 `166.43`。
- 已跑 SAME official-consistency 20-step smoke：`tmp/same_flow_debug_train/official_consistency_smoke20_rerun/train_log.json`。step1/5/10/15/20 loss 分别约 `182.92/378.69/101.73/855.00/72.70`；低 sigma 样本权重大时 loss 会极端波动。
- 表示尺度不同，不能把裸 loss 绝对值当完全同分布：
  - official Music2Latent overfit wav：representation shape `[1,2,1024,184]`，std `0.3483`，元素数 `376,832`。
  - SAME-flow debug：representation shape `[1,4,480,400]`，std `0.2886`，元素数 `768,000`。
  - official huber 的 `c=0.00054*sqrt(data_dim)` 和 diff norm 都受 `data_dim` 影响，因此 SAME 的 loss 尺度天然受目标 `48k stereo/hop240` 表示影响。
- 可比较项应是：
  - 同一 official loss 公式下的 loss 趋势和 sigma-bucket 条件趋势；
  - `sigma_low/high`、`representation_std`、`latent_std` 同步记录；
  - direct `sigma=1/10/80` 的 MR-STFT/log-mel 指标。
- 不能比较项：
  - `max_denoise + MSE` debug loss 与 official consistency loss；
  - 不同 representation 维度下的裸 loss 绝对值，除非另做归一化或把 SAME 输入表示改成 official mono/hop512 parity 配置。

### 2026-06-19：按已验证小 Music2Latent baseline 做 parity 排查

用户修正 baseline 定义：当前用于排查的 baseline 不是 Music2Latent 官方默认大模型/800k 调度，而是已经试听和指标确认“可重建但有底噪”的小 fast-overfit 配置：

- `tmp/original_music2latent_official_overfit10_fast_config.py`
- 容量：`base_channels=32,bottleneck_base_channels=256,cond_channels=128,heads=4,num_bottleneck_layers=2,bottleneck_channels=64`
- 训练调度：`lr=3e-4,total_iters=3000,warmup_steps=100,final_lr=1e-5,ema_momentum=0.99`
- 不改官方目标函数、sigma 采样、STFT 公式、consistency pair loss。

为把差异降到最低，新增 SAME-flow debug parity 配置：

- `scripts/configs/experiment/same-flow-debug-m2l-parity.yaml`
- 数据 cache：`tmp/same_flow/m2l_fast_baseline_cache_48000_mono_34304/index.jsonl`
- cache 从 `tmp/original_music2latent_official_overfit10_wav/*.wav` 直接取 48k mono、34304 samples，不重采样；这匹配官方 dataloader 实际读到的 waveform tensor。`hparams.sample_rate=44100` 在官方 baseline 中主要用于保存/评估，不参与 STFT frame 计算。
- STFT 口径：`hop=512,fac=4,data_channels=2,center_pad=false`，因此 `34304 samples -> representation [B,2,1024,64] -> latent [B,64,8]`，与 Music2Latent 小 baseline 对齐。
- 训练 wrapper 继续使用 official huber、official lognormal sigma、RAdam + fast cosine/warmup。

为支持 no-center-padding parity，`scripts/same_flow_debug.audio.AudioProcessor` 新增 `center_pad` 开关，默认保持旧行为；`same-flow-debug-m2l-parity.yaml` 设 `center_pad=false`。`debug_train.py` 和 wrapper 的裁剪逻辑也按 `center_pad` 分流：center pad 旧公式是 `frames*hop`，official no-pad 公式是 `frames*hop + (fac-1)*hop`。parity smoke：

- audio `[1,34304]` std `0.32984`
- representation `[1,2,1024,64]` std `0.46674`
- latent `[1,64,8]` std 约 `0.31`
- `training_step` finite。

逐层 trace 脚本：

- `scripts/same_flow_debug/compare_music2latent_trace.py`
- 读取同一条 parity audio、同一个 `sigma`、同一个 noise seed，手工按官方卷积 forward 和 SAME forward 记录每个等价 stage 的 `shape/mean/std/rms`。
- random/init trace 输出：`tmp/same_flow_debug_trace/m2l_fast_parity_random_sigma10_item0_zerohead.json`

trace 发现并修正的关键不等价：

- 官方 Music2Latent 最终 `conv_out` 是 zero-init，random/init 时 residual head RMS 为 `0`，`edm.output` 只来自 `c_skip * noisy`，sigma10 输出 RMS `0.0249946`。
- SAME-flow 原先最终 `output_proj` 用 `SameFlowBlock`，当 `in_channels==out_channels` 时有 residual identity；random/init 的 residual representation RMS 曾达到 `0.2085`，不是 baseline 等价行为。
- 已在 `scripts/same_flow_debug.models` 增加 `ZeroTokenProjection`，作为无 residual、zero-init 的 final token head；parity 配置设 `direct_patch_output=true`，避免 zero token 再经过 `patch_decode` 随机 bias 生成非零频谱残差。
- 修正后 random/init trace：SAME 和 Music2Latent 的 representation 完全一致，`edm.output` RMS 都是 `0.0249946`，final residual head 都是 `0`。

修正后的 random/init 中间层仍有结构差异，后续应重点排查：

| stage | Music2Latent RMS | SAME RMS | ratio |
| --- | ---: | ---: | ---: |
| `representation` | 0.4667 | 0.4667 | 1.00 |
| `encoder.latent` | 0.3116 | 0.3146 | 1.01 |
| `denoiser.input_proj` | 0.582 | 0.219 | 0.38 |
| `down.0.0.mixed` | 0.4185 | 0.1688 | 0.40 |
| `down.1.0.block` | 0.1369 | 0.1286 | 0.94 |
| `up.0.0.block` | 0.0977 | 0.2917 | 2.99 |
| `up.1.transition` | 0.0375 | 0.3410 | 9.10 |
| `final.norm_act` | 0.5869 | 0.5728 | 0.98 |
| `edm.output` | 0.02499 | 0.02499 | 1.00 |

解释：STFT 输入、latent 和初始化 EDM 边界已经对齐；当前真正的结构差异在 SAME 把频率 patch 压成 token 后，早期 denoiser 输入能量低于卷积 baseline，up path transition 又明显高于 baseline。下一步不要再改 loss，应在 parity 配置上训练 SAME zero-head，并与 `tmp/original_music2latent_official_short_eval_fast/step_001500.pt` 做同口径 trace/指标比较。

当前运行：

```sh
python -m scripts.same_flow_debug.debug_train \
  --experiment same-flow-debug-m2l-parity \
  --output-dir tmp/same_flow_debug_train/m2l_parity_zerohead_1500 \
  --max-steps 1500 --batch-size 4 --save-every 500 --log-every 50 \
  --probe-every 250 --probe-sigma 10 --device cuda:0
```

`debug_train.py` 的 probe 指标已修复 mono audio 处理。1-step probe smoke：loss `143.99`、lr `3e-6`、probe RMS `0.0006`，符合 zero-head + warmup 初始行为。

`m2l_parity_zerohead_1500` 已完成：

- checkpoint：`tmp/same_flow_debug_train/m2l_parity_zerohead_1500/last.ckpt`
- step1500 probe：loss `176.20`、probe RMS `0.0020`、MRSTFT-SC `0.997`、logmel L1 `3.741`，仍接近零输出。
- trace：`tmp/same_flow_debug_trace/m2l_fast_parity_official1500_same1500_sigma10_item0.json`

与 official small baseline step1500 的同音频 sigma10 trace：

| stage | official step1500 RMS | SAME zerohead step1500 RMS | ratio |
| --- | ---: | ---: | ---: |
| `encoder.latent` | 0.57445 | 0.20313 | 0.35 |
| `denoiser.input_proj` | 0.57400 | 0.12940 | 0.23 |
| `down.0.0.mixed` | 0.42491 | 0.07816 | 0.18 |
| `down.1.0.block` | 0.33606 | 0.16415 | 0.49 |
| `up.0.0.block` | 0.56466 | 0.27933 | 0.49 |
| `up.1.transition` | 0.46418 | 0.43664 | 0.94 |
| `final.norm_act` | 0.60147 | 0.43179 | 0.72 |
| `final.residual_head` | 0.45351 | 0.06677 | 0.15 |
| `edm.output` | 0.23268 | 0.03945 | 0.17 |

梯度检查：

- zero-head 初始时上游梯度为 0 是预期，因为 final projection 权重为 0；但 `output_proj.proj.weight grad=3.43`、bias grad `0.42`，说明没有断梯度。
- 1500 step 后 final residual head 仍只有 official 的约 `15%`，而 `up.1.transition` 已接近 official，说明主要问题不是 loss 或全网不学习，而是 patch-token 到频谱 residual head 的信息/尺度效率弱。

当前更合理的下一步结构对照：

- 在 parity 配置中只改 `model.patch_channels=64`，因为当前 parity 的 `patch_dim = data_channels * freq_patch_size = 2 * 32 = 64`，而默认 `patch_channels=32` 会先把每个频率 patch 从 64 维压到 32 维；官方卷积 baseline 没有这个早期 patch 信息瓶颈。
- 正在运行：

```sh
python -m scripts.same_flow_debug.debug_train \
  --experiment same-flow-debug-m2l-parity \
  --output-dir tmp/same_flow_debug_train/m2l_parity_zerohead_patch64_1500 \
  --max-steps 1500 --batch-size 4 --save-every 500 --log-every 50 \
  --probe-every 250 --probe-sigma 10 --device cuda:0 \
  --override model.patch_channels=64
```

补充结构并行实验：

- `raw_patch_tokens=true`：让 patch denoiser 直接吃 raw STFT patch 值 `[patch_dim=64]`，绕过 `patch_embed`；仍用 patch 序列和 direct patch zero head。
- `raw_patch_tokens=true + transformer_depth=2`
- `raw_patch_tokens=true + sliding_window=[16,16]`
- `use_patch_denoiser=false + direct_frame_output=true`：不拆 patch，每个 STFT frame 一个 token，zero head 直接输出整帧 `2*1024` 频谱 residual。

早期结果：

| run | step | probe RMS | MRSTFT-SC | logmel L1 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `m2l_parity_rawpatch_1500` | 250 | 0.0015 | 0.999 | 3.886 | 基本仍近零，已停 |
| `m2l_parity_rawpatch_depth2_1500` | 250 | 0.0015 | 0.999 | 3.892 | 基本仍近零，已停 |
| `m2l_parity_rawpatch_window16_1500` | 500 | 0.0012 | 1.000 | 4.098 | 基本仍近零，已停 |
| `m2l_parity_framedirect_1500` | 500 | 0.0360 | 0.944 | 3.114 | 明显开始学 |
| `m2l_parity_framedirect_1500` | 1000 | 0.1253 | 0.813 | 2.895 | 持续增强 |
| `m2l_parity_framedirect_1500` | 1500 | 0.1421 | 0.734 | 2.836 | 当前最有效结构 |
| `m2l_parity_framedirect_depth2_1500` | 1000 | 0.1349 | 0.798 | 2.830 | 与 depth1 接近，仍在跑 |

正确加载 frame-direct checkpoint 的 trace：

- `tmp/same_flow_debug_trace/m2l_fast_parity_official1500_framedirect1500_sigma10_item0_correct.json`

同 official small baseline step1500 对比：

| stage | official step1500 RMS | frame-direct step1500 RMS | ratio |
| --- | ---: | ---: | ---: |
| `encoder.latent` | 0.57445 | 0.40989 | 0.71 |
| `denoiser.input_proj` | 0.57400 | 0.70835 | 1.23 |
| `down.0.0.mixed` | 0.42491 | 0.93749 | 2.21 |
| `down.1.0.block` | 0.33606 | 0.49642 | 1.48 |
| `up.0.0.block` | 0.56466 | 0.65175 | 1.15 |
| `up.1.transition` | 0.46418 | 0.91443 | 1.97 |
| `final.norm_act` | 0.60147 | 0.70423 | 1.17 |
| `final.residual_head` | 0.45351 | 0.65022 | 1.43 |
| `edm.output` | 0.23268 | 0.32691 | 1.40 |

结论：主要瓶颈不是 patch_embed 的 64->32 通道宽度，而是 patch-split denoiser 把频率 patch 拆成独立序列后，official consistency 很难快速学出全频 residual。frame-direct 让模型直接按整帧输出频谱 residual，能量学习速度接近/超过卷积 baseline，但音频 probe 指标仍落后，后续应围绕 frame-direct 跑完整 3000 step 并分析频谱误差，而不是继续放大 patch denoiser。

正在运行：

```sh
python -m scripts.same_flow_debug.debug_train \
  --experiment same-flow-debug-m2l-parity \
  --output-dir tmp/same_flow_debug_train/m2l_parity_framedirect_3000 \
  --max-steps 3000 --batch-size 4 --save-every 500 --log-every 100 \
  --probe-every 500 --probe-sigma 10 --device cuda:3 \
  --override model.use_patch_denoiser=false \
  --override model.direct_patch_output=false \
  --override model.direct_frame_output=true
```

2026-06-19 补充：frame-direct 1500 可听样例已生成。

评估命令使用 `tmp/same_flow_debug_train/m2l_parity_framedirect_1500/last.ckpt`，并显式保持训练结构：

```sh
python -m scripts.same_flow_debug.eval_checkpoint \
  --experiment same-flow-debug-m2l-parity \
  --checkpoint tmp/same_flow_debug_train/m2l_parity_framedirect_1500/last.ckpt \
  --variant raw \
  --output-dir tmp/same_flow_debug_listen/m2l_parity_framedirect_1500_raw_step1 \
  --max-items 10 \
  --denoising-steps 1 \
  --sampling-mode deterministic \
  --direct-sigma 10 \
  --device cuda:0 \
  --override model.use_patch_denoiser=false \
  --override model.direct_patch_output=false \
  --override model.direct_frame_output=true
```

修复点：

- `eval_checkpoint.py` 现在会把 mono `[T]` 统一成 `[1,T]` 后保存和算指标，避免 `torchaudio.save` 与 STFT 指标形状错误。
- parity eval 不再调用通用 `model.encode()`，而是 `audio_processor.to_representation_encoder(source.unsqueeze(0)) -> model.encoder(representation)`；原因是 no-center-pad parity 的训练 latent 是 `[B,64,8]`，通用 `encode()` 的 padding 公式会误扩成 `[B,64,9]`。

结果：

| path | corr mean | SNR mean | MRSTFT-SC | logmel L1 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1-step deterministic decode | 0.4423 | 0.30 dB | 0.941 | 2.415 | 可听，但仍有失败样本 |
| direct sigma=10 denoise | 0.5513 | 0.66 dB | 1.087 | 2.648 | 相关性更高，频谱指标不一定更好 |

音频输出：

- 全量 10 条：`tmp/same_flow_debug_listen/m2l_parity_framedirect_1500_raw_step1/`，每条包含 `source.wav`、`reconstruct_raw.wav`、`direct_sigma10_raw.wav`。
- 拼接听感样例：`tmp/same_flow_debug_listen/assembled/overview_src_recon_direct_selected.wav`。
- 单条代表样例：`tmp/same_flow_debug_listen/assembled/item007_corr0.803_direct0.862_src_recon_direct.wav`、`item002_corr0.739_direct0.796_src_recon_direct.wav`、`item001_corr0.368_direct0.697_src_recon_direct.wav`、`item006_corr0.013_direct0.014_src_recon_direct.wav`、`item008_corr0.016_direct0.011_src_recon_direct.wav`。

2026-06-19 继续：`m2l_parity_framedirect_3000` 已自然完成并评估。

checkpoint：`tmp/same_flow_debug_train/m2l_parity_framedirect_3000/last.ckpt`。

10 条 1-step deterministic decode：

| corr mean | SNR mean | MRSTFT-SC | logmel L1 | prediction RMS | source RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7370 | 6.77 dB | 0.461 | 2.299 | 0.1107 | 0.1322 |

`direct_sigma=10` 同批：`direct_corr_mean=0.7368,direct_SNR=5.66dB`，相关性不再优于 deterministic decode，且 MRSTFT 更差；当前听感主样例优先用普通 `reconstruct_raw.wav`。

可听文件：

- 全量 10 条：`tmp/same_flow_debug_listen/m2l_parity_framedirect_3000_last_raw_step1/`。
- 拼接总览：`tmp/same_flow_debug_listen/assembled_3000_last/overview_3000_last_src_recon_direct.wav`。
- 好样本：`tmp/same_flow_debug_listen/assembled_3000_last/item001_corr0.978_snr13.4_src_recon_direct.wav`、`item000_corr0.960_snr10.7_src_recon_direct.wav`、`item004_corr0.963_snr9.8_src_recon_direct.wav`。
- 失败样本仍集中在 `item006_corr0.031_snr-0.1_src_recon_direct.wav` 和 `item008_corr0.020_snr-1.5_src_recon_direct.wav`，后续需要查数据内容/latent/频谱，而不是只看平均指标。

2026-06-19 结构对照与静音样本修正。

新增代码：

- `scripts/same_flow_debug.models.SpectralFrameAdapter.to_raw_frame_tokens()` 和 `Music2LatentSameFlow(raw_frame_tokens=true)`，用于 `direct_frame_output=true` 时让 denoiser 直接吃 raw full-frame token `[B, data_channels*freq_bins=2048, T]`，去掉 `to_tokens()` 的 `2048 -> frame_channels=512` 输入压缩。
- `scripts/same_flow_debug/eval_checkpoint.py` 增加 `--quiet-rms-threshold`，默认 `0.01`，summary 里写 `nonquiet_*` 和 `quiet_*` 指标。原因：当前 10 条 parity eval 中 item006/item008 是近静音片段，source RMS 只有约 `0.0007/0.0002`，corr 对这些样本没有稳定意义。

1500 step 结构对照，同一 parity 数据、同一 official consistency loss、同一 deterministic 1-step eval：

| run | corr mean | SNR mean | MRSTFT-SC | logmel L1 | nonquiet corr | nonquiet SNR | nonquiet MRSTFT | quiet pred RMS | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `m2l_parity_framedirect_1500` | 0.4423 | 0.30 | 0.941 | 2.415 | 0.5492 | 2.08 | 0.682 | 0.000643 | 原始主线 1500 |
| `m2l_parity_rawframe_1500` | 0.5099 | 0.82 | 0.877 | 2.451 | 0.6357 | 2.72 | 0.631 | 0.000640 | raw-frame 有早期收益，继续到 3000 |
| `m2l_parity_framedirect_p64_fc2048_1500` | 0.3566 | -0.13 | 0.878 | 2.383 | 0.4394 | 1.22 | 0.719 | 0.000519 | 不继续 |
| `m2l_parity_framedirect_p64_fc1024_1500` | 0.3670 | -0.17 | 0.937 | 2.413 | 0.4555 | 1.35 | 0.738 | 0.000582 | 不继续 |
| `m2l_parity_framedirect_3000` | 0.7370 | 6.77 | 0.461 | 2.299 | 0.9149 | 8.67 | 0.363 | 0.000128 | 当前最强 baseline |

结论：

- p64/fc1024/fc2048 不是正确方向，虽然扩大 frame adapter 容量，但没有改善训练效率。
- raw-frame 去掉输入压缩后 1500 step 优于原始 1500，值得跑到 3000 观察是否超过原始 frame-direct。
- 原始 frame-direct 3000 已在非静音样本上接近可用重建；失败样本主要是静音/极低能量，后续报告必须同时看 `nonquiet_*` 与 `quiet_prediction_rms`，不能只看全量 corr。

正在运行：

```sh
python -m scripts.same_flow_debug.debug_train \
  --experiment same-flow-debug-m2l-parity \
  --output-dir tmp/same_flow_debug_train/m2l_parity_framedirect_6000 \
  --max-steps 6000 --batch-size 4 --save-every 1000 --log-every 100 \
  --probe-every 1000 --probe-sigma 10 --device cuda:0 \
  --override model.use_patch_denoiser=false \
  --override model.direct_patch_output=false \
  --override model.direct_frame_output=true \
  --override wrapper.lr_schedule_total_steps=6000 \
  --override wrapper.consistency_step_total_steps=6000
```

```sh
python -m scripts.same_flow_debug.debug_train \
  --experiment same-flow-debug-m2l-parity \
  --output-dir tmp/same_flow_debug_train/m2l_parity_rawframe_3000 \
  --max-steps 3000 --batch-size 4 --save-every 500 --log-every 100 \
  --probe-every 500 --probe-sigma 10 --device cuda:1 \
  --override model.use_patch_denoiser=false \
  --override model.direct_patch_output=false \
  --override model.direct_frame_output=true \
  --override model.raw_frame_tokens=true
```

2026-06-19 `scripts/same_flow` baseline 收敛：

- 生产 baseline 类改名为 `scripts.same_flow.models.SameFlowConsistencyAutoencoder`，不再用 `Music2LatentSameFlow` 命名；`scripts/configs/experiment/same-flow-overfit10.yaml` 和新增 `same-flow-baseline-m2l-parity.yaml` 都指向新类名。
- 主模型固定为已经验证通的 frame-direct full-frame residual head：`SpectralFrameAdapter.to_tokens()` 负责 `[B,C,F,T] -> [B,frame_channels,T]`，denoiser 最后 `ZeroTokenProjection(input_channels, data_channels*freq_bins)`，再 `frame_values_to_representation()` reshape 回 `[B,C,F,T]`。不再使用旧的 `frame_unproj + patch_decode` 从 frame token 还原频谱。
- 从 `scripts/same_flow.wrapper.SameFlowTrainingWrapper` 和 YAML 里删除无效 0 权重分支：`denoising_anchor_weight`、`high_freq_anchor_weight`、`high_freq_anchor_start_bin`、`latent_prior_loss_weight`。baseline 只保留 official consistency pair loss、sigma schedule、LR schedule 和 EMA。
- 删除 main baseline 中 unused latent-prior 模块：`prior_norm/prior_proj/latent_prior()`。这也是 DDP 报错的原因之一。
- `scripts/same_flow.audio.AudioProcessor` 增加 `center_pad`，`wrapper._crop_to_valid_stft_length()` 支持 no-center-pad parity 公式；`eval_checkpoint.py` 支持 mono 保存、representation 直接 encoder latent、`quiet/nonquiet` 汇总指标。

用户启动 `same-flow-overfit10` 时 DDP 报错：

```text
parameters that were not used in producing the loss
```

本地 backward 检测确认 unused 参数为：

- `model.adapter.frame_unproj.*`
- `model.adapter.patch_decode.*`
- `model.prior_norm.*`
- `model.prior_proj.*`

修复后验证：

- `python -m py_compile scripts/same_flow/*.py` 通过。
- baseline parity smoke：`source (1,34304) -> rep (1,2,1024,64) -> latent (1,64,8) -> out (1,2,1024,64)` 通过。
- `same-flow-overfit10` local backward unused 检测：`unused_count 0`。
- 真实 Lightning/DDP 单卡 2-step smoke 通过：

```sh
python -m scripts.train experiment=same-flow-overfit10 \
  trainer.max_steps=2 trainer.devices=1 trainer.num_nodes=1 \
  callbacks.demo_callback.demo_every=999999 \
  paths.output_dir=tmp/same_flow_train_smoke
```

模型参数从约 `13.4M` 降到 `9.7M`，删除的是确实不参与当前 baseline loss 的旧解码/latent-prior分支。

2026-06-19 `same_flow` debugfaithful parity：

用户发现整理版 `same-flow-baseline-m2l-parity` 的 Lightning 500-step 输出 RMS 明显低于旧 `scripts/same_flow_debug` 成功 baseline。排查结论：

- `scripts/same_flow.debug_train` 与旧 `scripts/same_flow_debug.debug_train` 在模型结构、loss、sigma 序列和 LR 上可以复现旧 debug 口径；500 step 对齐结果：
  - production debugfaithful：`corr_mean=0.1324`、`prediction_rms_mean=0.02233`、`direct_corr_mean=0.2436`、`nonquiet_direct_corr_mean=0.3052`。
  - old debug `m2l_parity_framedirect_3000/step_000500.ckpt`：`corr_mean=0.1303`、`prediction_rms_mean=0.02198`、`direct_corr_mean=0.2627`、`nonquiet_direct_corr_mean=0.3271`。
- 真正差异是 Lightning 会按官方 Music2Latent `get_step_schedule(self.it)` 推进 consistency step；到 step500 时 step 从 `0.1` 衰减到约 `0.046`，早期输出更保守。旧 debug 成功实验因为 debug loop 没推进 LightningModule `global_step`，实际一直用最大 step `0.1`，更像强去噪训练，因此 RMS 拉得更快。
- 为了显式复刻旧 debug 成功口径，新增 `scripts/configs/experiment/same-flow-baseline-m2l-parity-debugfaithful.yaml`：仍使用整理版 `scripts.same_flow.models.SameFlowConsistencyAutoencoder`，但设置 `wrapper.consistency_step_schedule: constant`、`consistency_step: 0.1`。这个配置用于结构回归测试，不代表最终官方 schedule。
- `scripts/same_flow/wrapper.py` 和 `scripts/same_flow_debug/wrapper.py` 现在会日志记录 `consistency/step_size`；两个 `debug_train.py` 支持 `_debug_global_step_override`，后续调试脚本能选择与官方 schedule 对齐，不再隐式固定 step0。
- 500-step 复刻试听：
  - assembled: `tmp/same_flow_listen/assembled_baseline_m2l_parity_debugfaithful_500/overview_500_src_recon_direct.wav`
  - per-item: `tmp/same_flow_listen/baseline_m2l_parity_debugfaithful_500_raw_step1/`

当前正在跑 3000-step 复刻：

```sh
python -m scripts.same_flow.debug_train \
  --experiment same-flow-baseline-m2l-parity-debugfaithful \
  --output-dir tmp/same_flow_train/baseline_m2l_parity_debugfaithful_3000 \
  --max-steps 3000 --batch-size 4 --save-every 500 --log-every 100 \
  --device cuda:1
```

2026-06-19 正式 `same-flow.yaml`：

- 新增正式入口 `scripts/configs/experiment/same-flow.yaml`，不再覆盖旧 `scripts/configs/experiment/same.yaml`；旧 `same.yaml` 仍属于 `scripts.same` autoencoder/GAN 路线。
- `same-flow.yaml` 使用整理版 `scripts.same_flow.models.SameFlowConsistencyAutoencoder`，并沿用 debugfaithful 成功逻辑：`consistency_step_schedule=constant`、`consistency_step=0.1`、1-step deterministic demo、RAdam + Music2Latent cosine LR。
- STFT 口径按已成功 parity 实验保持：`sample_rate=44100`、`hop=512`、`audio_processor.center_pad=false`、`alpha_rescale=0.65`、`beta_rescale=0.34`、`fac=4`。正式数据使用双声道，因此 `dataset.num_channels=2`、`model.data_channels=4`。
- 数据从 overfit10 切到全量 raw index：`tmp/same/netease_spider_audio.index.list`，`max_duration_seconds=300`，逻辑长度 `10000000`。
- 模型按用户要求扩大到约 579M：`base_channels=128`、`frame_channels=1024`、`bottleneck_base_channels=1024`、`cond_channels=512`、`transformer_depth=3`、`dim_heads=64`、`layers_list=[1,1,1,1,1]`、`layers_list_encoder=[1,1,1,1,1]`。实例化验证参数量 `578.954M`。
- 当前样本长度先设 `sample_size=882000`（44.1k 下 20 秒），batch size `1`，粗略 latent 时间约 `215`。下一步需要单卡 smoke 观察峰值显存，再优先增加 `sample_size`，目标让单卡显存接近 80%。
- 为大模型 demo 降低 OOM 风险，`scripts.same_flow.wrapper.SameFlowTrainingCallback` 和 debug 副本现在会在 demo 前、raw/EMA demo 之间、每个样本保存后、以及 finally 收尾时执行 `gc.collect()` + `torch.cuda.synchronize()` + `torch.cuda.empty_cache()`；后续 demo 问题先查是否是 decode 峰值而不是训练 step 峰值。
