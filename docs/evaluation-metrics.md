# Mossland 任务评估指标调研报告

本报告针对 `scripts/mossland-codec/tasks.py` 里的 5 个任务整理评估指标、论文来源和可复用代码仓库。论文 PDF 已下载到 `docs/papers/`，索引见 `docs/papers/README.md`。

## 结论摘要

| 任务 | 首要评估 | 辅助客观指标 | 关键注意事项 |
| --- | --- | --- | --- |
| `reconstruct` | MUSHRA / MUSHRA-like 主观听测 | ViSQOLAudio、SI-SDR/SI-SNR、mel/STFT distance、FAD、码率/实时率 | Codec/reconstruction 的最终目标是听感；波形误差只能作诊断。 |
| `separate_vocals` | BSS Eval v4 / `museval` + 分离听测 | SDR/SIR/SAR/ISR、SI-SDR/SD-SDR、stem-sum consistency、PEASS 可选 | 当前 prepared stem 来自 RoFormer teacher；若无人工 stem，指标是 teacher-matching，不是真实分离上限。 |
| `separate_accompaniment` | 同上 | 同上 | accompaniment 往往比 vocals 更受 artifacts 和 residual vocal 影响，应单独报。 |
| `super_resolution` | MUSHRA / preference / MOS，按输入带宽分桶 | LSD、LSD-HF/LSD-LF、ViSQOLAudio、MRSTFT、HF energy、FAD | SNR 不足以衡量高频生成；AudioSR 明确观察到 LSD 与听感可不一致。 |
| `mono_to_stereo` | 生成式 stereo rendering 主观听测：naturalness、quality、spatialization、preference | fold-down 保真、normalized width、channel/IACC correlation、phase correlation、M/S FAD、IID/IPD/IC 统计、多样性 | 不是确定性映射。原始 stereo 只能作统计参考或上界样本，不能把 L/R waveform error 当主指标。 |

## 统一评估协议

1. 固定 evaluation manifest：记录 `path`、`task_id`、`start/end`、`sample_rate`、`low_sample_rate`（audio super-resolution）、`seed`（生成式任务）和输出文件路径。不要让随机 crop 或随机 task sampling 进入评估。
2. 统一采样率和声道：当前训练数据是 44.1 kHz stereo prepared mp3，模型参数里也有 48 kHz/44.1 kHz 历史。评估时必须显式写清楚重采样路径。
3. 做响度控制：DAC 论文评估前按 -24 dB LUFS 归一化训练片段；DiffStereo 也提醒响度会影响主观偏好。听测和 FAD 前应固定响度策略，并额外报告 peak clipping。
4. 每文件先算指标再聚合：不要把所有音频拼起来算全局误差。Separation 建议 track-wise median 后再跨曲目统计；audio super-resolution 和 reconstruction 建议按样本求均值/中位数并给 95% bootstrap CI。
5. 报告 baseline：每个任务都要有 trivial baseline，避免模型指标不可解释。建议包括 identity/low-pass/linear-upsample/mono-duplicate/RoFormer teacher 等。
6. 主观听测必须包含 anchor 和质量控制：MUSHRA 或 MUSHRA-like 使用 hidden reference、low anchor；若是 `mono_to_stereo`，可把 mono duplicate 作为低空间 anchor，把专业 stereo 作为参考上界，但问题描述应强调“空间/整体偏好”，不是“匹配唯一真值”。

## `reconstruct`

任务语义：`src == target`，更接近 neural audio codec / autoencoding reconstruction。模型既要保存内容，也要避免低码率或生成式 decoder 的听感 artifacts。

推荐指标：

- **MUSHRA / MUSHRA-like**：主指标。EnCodec 使用 hidden reference 和 low anchor，50 个 5 秒样本、每样本至少 10 个标注，并过滤不能识别 reference/anchor 的 annotator；SoundStream 和 DAC 也以 MUSHRA-like 听测为主要 codec 结果。
- **ViSQOLAudio**：开发和 ablation 指标。EnCodec、SoundStream、DAC 都使用 ViSQOL 或 ViSQOLAudio 做客观质量估计。ViSQOL v3 论文说明它是 full-reference/intrusive 质量估计工具，并且 Google 已开源 C++ 实现。
- **SI-SDR / SI-SNR**：诊断 waveform/phase 保真。DAC 明确把 SI-SDR 与 spectral metrics 一起使用，用来观察 phase reconstruction；但它不应替代主观听测。
- **mel distance / STFT distance / MRSTFT**：训练损失和诊断指标。DAC 同时报告 mel distance 与 STFT distance；EnCodec/SoundStream 的训练和 discriminator 也大量使用 multi-scale STFT/mel。
- **FAD / FD**：分布级听感补充。FAD 原论文认为 signal-level metrics 不一定预测音乐感知质量，FAD 比较 embedding 分布而非逐样本误差。注意 FAD 音乐评估论文指出 FAD 有 sample-size bias，样本数必须固定并尽量大，或报告 FAD-infinity 类校正。
- **码率、压缩率、codebook utilization、RTF**：如果 Mossland 后续把 codec latent 作为传输码流，必须按 bitrate/latent budget 作 rate-quality 曲线；没有稳定码流时先作为内部诊断。

建议报告：

- 每个 domain：music / vocals-heavy / instrumental / noisy source 分开报。
- 每个 latent budget 或采样步数：`MUSHRA`、`ViSQOLAudio`、`SI-SDR`、`mel/STFT distance`、`FAD`、`RTF encode/decode`。
- demo 音频：reference、model、anchor、baseline 同目录输出，便于 webMUSHRA 使用。

## `separate_vocals` 与 `separate_accompaniment`

任务语义：`src=mixture`，`target=vocals` 或 `target=accompaniment`。当前训练目标来自 `prepare_separation.py` 生成的 RoFormer prepared stems，而不是人工 multitrack stem。

推荐指标：

- **BSS Eval v4 / `museval`**：标准客观指标，报告 `SDR`、`SIR`、`SAR`、`ISR`。SiSEC 2018 发布 MUSDB18 和 Python BSS Eval v4，论文说明 BSS Eval 通过固定全曲 distortion filter 避免 v3 短窗口 time-varying filter 高估性能，并用 `pip install museval` 作为官方工具。
- **track-wise median SDR**：MUSDB/SiSEC 常用做法。对每首歌先聚合 frame/window，再跨曲目看分布；vocals 和 accompaniment 分开报。
- **SI-SDR / SD-SDR / SI-SDR improvement**：BSS Eval SDR 在单通道场景有已知失效案例。`SDR - Half-baked or Well Done?` 建议用更简单的 SI-SDR 或 scale-aware SD-SDR 作为补充。
- **PEASS OPS/TPS/IPS/APS**：可选感知客观指标。`source_sep_human_vs_metrics.pdf` 说明 PEASS 试图预测 Overall/Target/Interference/Artifacts perceptual scores，但同一论文也发现 BSS Eval 和 PEASS 与听测相关性不强，不能作为唯一依据。
- **分离听测 / MUSHRA-like**：按 Interference、Target distortion、Artifacts、Overall quality 四个维度评估。EUSIPCO 2016 源分离听测论文就是这样把 MUSHRA 改造成四段听测。
- **mixture consistency**：检查 `vocals + accompaniment` 与 `mixture` 的 `SI-SDR/LSD/peak`，以及 residual vocal in accompaniment。这个不是分离质量充分条件，但能抓出训练输出不守恒的问题。

Mossland 特别注意：

- 如果评估集仍来自 RoFormer prepared stems，结果应标为 `teacher-stem SDR` 或 `RoFormer-target SDR`。它衡量 Mossland 是否复现 teacher，不衡量 teacher 是否接近真实 multitrack。
- 若需要发表级分离质量，应另跑 MUSDB18-HQ / MUSDB18 test 或 MDX-style 数据，使用人工 stem 作为 reference。
- `separate_accompaniment` 的 target 是完整伴奏，不是 drums/bass/other 分 stem，因此 `SIR/SAR` 与 vocals-only 的解释不同：更应关注 vocal leakage、音乐整体 artifacts 和 mixture consistency。

## `super_resolution`

任务语义：从低采样率/低带宽版本恢复全带音乐。当前 `low_sample_rate` 可随机采样 `[8000, 40000]`，但 payload 不包含实际 rate 条件，所以评估 manifest 必须外部记录实际 degradation rate。

推荐指标：

- **LSD**：audio super-resolution 最常见客观指标。NU-Wave 和 NU-Wave 2 都使用 `SNR + LSD`；AERO 使用 `LSD + ViSQOL + MUSHRA`。
- **LSD-HF / LSD-LF**：NU-Wave 2 明确把高频生成和低频保持分开：`LSD-HF` 看新生成高频，`LSD-LF` 看输入低频是否被破坏。Mossland 应采用这个分解。
- **SNR / SI-SDR**：只作内容保真辅助。NU-Wave 2 明确指出多个工作认为 SNR 不能衡量高频生成，不适合作为 upsampling 主指标。
- **ViSQOLAudio**：AERO 用 audio mode ViSQOL 评价 MusDB setting；适合做全参考感知质量补充。
- **MUSHRA / preference / MOS**：必须作为关键评估。AudioSR 论文观察到某些 ESC-50 设置里 LSD 最好的 baseline 反而主观质量最差，说明 audio super-resolution 不能只看 LSD。
- **HF energy / spectral rolloff / bandwidth occupancy**：诊断模型是否真的补高频，还是只做平滑 upsample。可按 cutoff 以上频段统计 energy ratio、rolloff、spectral flatness。
- **FAD**：用于看生成结果是否接近全带音乐分布，特别是模型可能生成“合理但不逐样本匹配”的高频纹理时。

推荐分桶：

- 固定输入带宽：`8k -> full`、`12k -> full`、`16k -> full`、`24k -> full`、`32k -> full`、`40k -> full`。每桶单独报指标。
- Baseline：linear/sinc upsample、zero-fill STFT/iSTFT、AERO/NU-Wave 2/AudioSR（如果可运行）、原始低带宽输入。
- Anchor：听测中把低带宽输入或 3.5 kHz/7 kHz low-pass 作为 anchor。

## `mono_to_stereo`

任务语义需要按用户补充重新定义：它不是单声道到某个唯一双声道 reference 的确定性映射，而是 **generative stereo rendering / upmix**。同一个 mono 可以对应多种合理 stereo image，专业 stereo mix 也包含艺术选择。

论文依据：

- `parametric_stereo_generation.pdf` 明确说 stereo image 是 highly subjective construct，同一输入可有 many plausible stereo renditions，并指出 common error measurements may not be appropriate。
- DiffStereo 把 mono-to-stereo 定义成 end-to-end stereo audio generation，使用 objective + subjective metrics，但也承认生成模型的 L/R/M/S error 更像诊断。
- BinauralGrad、Beyond Mono to Binaural、Sep-Stereo、Diff-SAGe 等空间音频论文说明可借鉴 binaural/spatial audio 的空间感、方向感、STFT/phase 和 MOS 评估。

推荐主观指标：

- **Spatial MOS**：听众评价空间感、宽度、沉浸感。DiffStereo 和 BinauralGrad 都单独报告 spatial MOS。
- **Naturalness MOS / Quality MOS**：防止模型只追求夸张宽度而牺牲音质。DiffStereo 的结果显示 spatial MOS 与 naturalness/quality 可能 trade off。
- **Preference test**：Parametric stereo generation 使用 0-100 preference，并把 mono downmix 和 original stereo 作为边界。Mossland 可采用 A/B 或多刺激 preference。
- **MUSHRA-like stereo test**：推荐条件包括 `mono duplicate`、classical decorrelation/pseudo-stereo baseline、model outputs、professional stereo reference。评分问题要写成“整体 stereo rendering 偏好/空间自然度”，不要写成“还原参考 stereo”。

推荐客观指标：

- **fold-down content preservation**：把输出折叠为 `M=(L+R)/2`，与输入 mono 比较 `SI-SDR`、`LSD`、`ViSQOLAudio`。这是硬约束：生成 stereo 不应改变歌曲内容。
- **mono compatibility / cancellation check**：输出折叠回 mono 后不能明显相消。报告 `M` energy、side leakage、peak/clipping。
- **M/S 指标**：对 `M=(L+R)/2`、`S=(L-R)/2` 分别计算 `LSD`、`SI-SDR`、`FAD`。DiffStereo 同时报告 Mid/Side 和 Left/Right LSD、SI-SNR、FAD。
- **stereo width / side energy**：normalized width、side-to-mid energy ratio、inter-channel level balance。
- **channel correlation / IACC**：DiffStereo 报告 channel correlation；parametric stereo 文献使用 interchannel coherence/correlation。过低可能是假宽，过高接近 mono。
- **phase correlation / IPD / ITD**：DiffStereo 报告 phase correlation；parametric stereo 使用 interchannel phase/time differences。音乐 stereo 可先按频带统计 IPD 分布，不一定要估计物理 DOA。
- **IID/ILD + IC 统计**：Parametric stereo coding 经典参数是 interchannel intensity difference、interchannel time/phase difference、interchannel coherence。可对生成结果和真实 stereo corpus 比较分布，而不是逐样本求同。
- **distribution FAD**：对 L/R、M/S 或 stereo embedding 分别算 FAD。注意 FAD 是集合指标，不能解释单首歌好坏。
- **diversity**：同一 mono 生成多次时，统计 side channel / PS parameter / M/S spectrogram 的 pairwise distance；同时检查 fold-down preservation。Parametric stereo generation 的 `E_min` 可作为多样本 oracle 诊断，但不能作为主榜，因为它鼓励采样很多次贴近单个 reference。

不建议作为主指标：

- 单个 reference 的 L/R waveform L1、L2、SI-SDR 排名。它会惩罚合理但不同的 pan/width/phase 选择。
- 只最大化 channel decorrelation 或 width。过宽、相位乱、mono folding 失败的结果可能主观更差。

## 代码仓库与工具

| 用途 | 仓库 | 说明 |
| --- | --- | --- |
| EnCodec baseline / codec API | https://github.com/facebookresearch/encodec | EnCodec 论文官方代码和模型。 |
| DAC baseline | https://github.com/descriptinc/descript-audio-codec | DAC / improved RVQGAN 官方代码。 |
| ViSQOL | https://github.com/google/visqol | ViSQOL v3 开源实现；EnCodec 使用 recommended recipes。 |
| FAD | https://github.com/google-research/google-research/tree/master/frechet_audio_distance | Google Research FAD 官方实现路径。 |
| webMUSHRA | https://github.com/audiolabs/webMUSHRA | MUSHRA / MUSHRA-like 浏览器听测框架。 |
| MUSDB18 loader | https://github.com/sigsep/sigsep-mus-db | MUSDB18 Python parser/tools。 |
| museval / BSS Eval v4 | https://github.com/sigsep/sigsep-mus-eval | SiSEC/MUSDB 常用 separation 评估工具。 |
| NU-Wave | https://github.com/maum-ai/nuwave | Diffusion audio upsampling。 |
| NU-Wave 2 | https://github.com/maum-ai/nuwave2 | 支持不同输入采样率的 audio upsampling。 |
| AudioSR | https://github.com/haoheliu/versatile_audio_super_resolution | Versatile audio super-resolution 官方代码。 |
| AERO | https://github.com/slp-rl/aero | AERO 项目页链接的官方代码。 |
| DiffStereo | https://github.com/SAKi-77/DiffStereo | Mono-to-stereo diffusion transformer。 |
| BinauralGrad | https://github.com/microsoft/NeuralSpeech/tree/master/BinauralGrad | BinauralGrad 代码在 Microsoft NeuralSpeech monorepo。 |
| Sep-Stereo | https://github.com/SheldonTsui/SepStereo_ECCV2020 | Visually guided stereophonic audio generation。 |
| speechmetrics | https://github.com/aliutkus/speechmetrics | PESQ、STOI 等 speech/audio metrics 包装；BinauralGrad 引用。 |
| auraloss | https://github.com/csteinmetz1/auraloss | MRSTFT / spectral losses 与指标实现；BinauralGrad 引用。 |

## 建议的 Mossland Eval v0

### 数据集

- `reconstruct`：从 held-out prepared mixture 中抽 300 个 10 秒片段，按 genre/source metadata 若可用分层。
- `separate_*`：先用当前 prepared stems 做 teacher-matching；另建 MUSDB18-HQ/MUSDB18 小评估集作为真实 stem sanity check。
- `super_resolution`：每个 `low_sample_rate` bucket 至少 50 个 10 秒片段，固定 downsample/upsample 实现。
- `mono_to_stereo`：从真实 stereo corpus 折叠 mono，保留原 stereo 作为统计参考；每个 mono 至少生成 4 个 seed。

### 输出字段

每条评估记录建议写成 JSONL：

```json
{
  "task_id": "super_resolution",
  "source_path": "...",
  "start_seconds": 12.5,
  "duration_seconds": 10.0,
  "sample_rate": 44100,
  "low_sample_rate": 16000,
  "seed": 0,
  "prediction_path": "...",
  "reference_path": "...",
  "metrics": {
    "lsd": 0.0,
    "lsd_hf": 0.0,
    "visqol_audio": 0.0
  }
}
```

### 汇总表

- 每个 `task_id` 单独表。
- audio super-resolution 按 `low_sample_rate` 分桶。
- Stereo 按 `seed` 和聚合后都报：content preservation、spatial stats、distribution stats、diversity。
- Separation 同时报 vocals/accompaniment 和 macro average，不用一个均值掩盖某个 stem 的失败。

## 主要论文来源

- EnCodec: High Fidelity Neural Audio Compression, `docs/papers/encodec.pdf`, https://arxiv.org/abs/2210.13438
- SoundStream: An End-to-End Neural Audio Codec, `docs/papers/soundstream.pdf`, https://arxiv.org/abs/2107.03312
- DAC: High-Fidelity Audio Compression with Improved RVQGAN, `docs/papers/dac_rvqgan.pdf`, https://arxiv.org/abs/2306.06546
- ViSQOL v3, `docs/papers/visqol_v3.pdf`, https://arxiv.org/abs/2004.09584
- Fréchet Audio Distance, `docs/papers/fad_original.pdf`, https://arxiv.org/abs/1812.08466
- Frechet Audio Distance for Music Evaluation, `docs/papers/fad_music_eval.pdf`, https://arxiv.org/abs/2311.01616
- SiSEC 2018 / MUSDB18 / BSS Eval v4, `docs/papers/sisec2018.pdf`, https://arxiv.org/abs/1804.06267
- BSS Eval original, `docs/papers/bss_eval_vincent.pdf`, https://hal.science/inria-00544230
- SDR - Half-baked or Well Done?, `docs/papers/sdr_half_baked.pdf`, https://arxiv.org/abs/1811.02508
- Are objective quality measures suited to evaluate audio source separation?, `docs/papers/source_sep_human_vs_metrics.pdf`, https://eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570256073.pdf
- NU-Wave, `docs/papers/nuwave.pdf`, https://arxiv.org/abs/2104.02321
- NU-Wave 2, `docs/papers/nuwave2.pdf`, https://arxiv.org/abs/2206.08545
- AudioSR, `docs/papers/audiosr.pdf`, https://arxiv.org/abs/2309.07314
- AERO, `docs/papers/aero.pdf`, https://arxiv.org/abs/2211.12232
- DiffStereo, `docs/papers/diffstereo.pdf`, https://www.isca-archive.org/interspeech_2025/zhang25q_interspeech.html
- Mono-to-stereo through parametric stereo generation, `docs/papers/parametric_stereo_generation.pdf`, https://archives.ismir.net/ismir2023/paper/000035.pdf
- BinauralGrad, `docs/papers/binauralgrad.pdf`, https://arxiv.org/abs/2205.14807
- Beyond Mono to Binaural, `docs/papers/beyond_mono_binaural.pdf`, https://openaccess.thecvf.com/content/WACV2022/html/Parida_Beyond_Mono_to_Binaural_Generating_Binaural_Audio_From_Mono_Audio_WACV_2022_paper.html
- Sep-Stereo, `docs/papers/sep_stereo.pdf`, https://arxiv.org/abs/2007.09902
- Diff-SAGe, `docs/papers/diff_sage.pdf`, https://arxiv.org/abs/2410.11299
- webMUSHRA, `docs/papers/webmushra.pdf`, https://doi.org/10.5334/jors.187
