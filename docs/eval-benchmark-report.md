# Mossland 评估复现报告

本报告跟踪 `scripts/mossland-codec/eval_benchmark/` 对 `scripts/mossland-codec/tasks.py` 五个任务的指标复现和 Mossland checkpoint 评估状态。当前是阶段性报告，不是最终全量结论。

## 报告口径

- 所有 Mossland 与 baseline 对比表只使用本仓库本地推理产物和统一评测 pipeline 重新计算的指标，不照搬论文表格数值。
- 论文复现状态单独标注：只有官方 checkpoint、官方/论文数据集、推理参数、采样率和指标后端都对齐时，才标为“论文协议复现”；否则标为“官方模型工程评测”或“未复现”。
- 当前 MusicCaps 结果基于第三方 HF 音频镜像 `mahendra0203/musiccaps_processed_full`，不是 Google 官方 MusicCaps 音频发布；因此所有 MusicCaps-HF 表都只能作为工程评测。
- `fad_clap`、`fad_vggish`、ViSQOL、museval 均来自本地实际计算。`fad_vggish` 当前后端是 `torchvggish_raw`，不是 Google TF/Beam 官方 FAD 脚本。

## 当前本地实测汇总

本节是 2026-06-14 的阶段性汇总，所有数值均来自本地推理产物和 `scripts/mossland-codec/eval_benchmark/` 统一 pipeline。`MusicCaps-HF` 表使用第三方 Hugging Face 音频镜像，不是 Google 官方 YouTube 音频完整复现；`MUSDB18-HQ 10s` 是每首 test track 的非静音 10 秒窗口，不是 SiSEC full-track 官方口径。

### Reconstruction / Codec

| 模型 | 数据 | 样本数 | FAD-CLAP ↓ | FAD-VGGish ↓ | SI-SDR ↑ | SNR ↑ | LSD ↓ | MRSTFT ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mossland `ckpt/mossland-codec0613` | MusicCaps-HF full | 5355 | 0.361294 | 0.871600 | -14.0029 | -1.34065 | 29.7523 | 708.206 mean / 0.561047 median |
| EnCodec 48 kHz 24 kbps | MusicCaps-HF full | 5355 | 0.0408916 | 0.981195 | 10.8863 | 11.2612 | 16.7002 | 0.282905 |
| DAC 44.1 kHz 8 kbps | MusicCaps-HF full | 5355 | 0.075158 | 0.713645 | 9.72614 | 4.72790 | 22.4124 | 0.557487 |
| SNAC 44 kHz | MusicCaps-HF full | 5355 | 0.0563409 | 1.64879 | 2.13932 | 4.23513 | 17.1182 | 0.462963 |
| WavTokenizer music 75-token | MusicCaps-HF full | 5355 | 0.0764699 | 1.61131 | -3.28095 | 1.72287 | 18.1491 | 0.541830 |
| CoDiCodec official checkpoint | MusicCaps-HF full | 5355 | 0.0915413 | 0.496719 | -1.12868 | 2.37269 | 20.8388 | 0.490066 |

Mossland 已完成 5355 条 full ViSQOL：MOS-LQO mean `2.97158`、median `2.78838`。SNAC、WavTokenizer 和 CoDiCodec 均已补 5355 条 full FAD/距离指标。已完成的 100 条 ViSQOL：SNAC `4.36742`，WavTokenizer `4.14317`，CoDiCodec `4.17106`；这些 100 条 ViSQOL 暂不混作 full 5355 统计。

#### SemantiCodec 100 条 rate-quality 阶段结果

SemantiCodec 使用官方 repo `tmp/eval_baseline_refs/semanticodec/SemantiCodec-inference` 与官方 Hugging Face weights，本表是 MusicCaps-HF 前 100 条 reconstruction 工程评测。当前已补 CLAP/VGGish/距离指标，暂不启动 ViSQOL，避免和 full codec ViSQOL 后台任务抢 CPU。

| 模型 | token-rate | 样本数 | FAD-CLAP ↓ | FAD-VGGish ↓ | SI-SDR ↑ | SNR ↑ | LSD ↓ | MRSTFT ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SemantiCodec official | 25 | 100 | 0.118665 | 3.53085 | -37.4404 | -2.59877 | 10.6071 | 0.630190 |
| SemantiCodec official | 50 | 100 | 0.135771 | 3.73924 | -34.4658 | -2.49581 | 10.0890 | 0.576908 |
| SemantiCodec official | 100 | 100 | 0.0838395 | 2.69508 | -35.1448 | -2.58977 | 9.98482 | 0.542909 |

#### EnCodec full rate-quality 阶段结果

EnCodec 使用官方 `encodec_model_48khz()`，本表是 MusicCaps-HF full 5355 条 reconstruction 的同一统一 pipeline CLAP/VGGish/距离评测。`24 kbps` 行来自主 full reconstruction 表。

| 模型 | bandwidth | 样本数 | FAD-CLAP ↓ | FAD-VGGish ↓ | SI-SDR ↑ | SNR ↑ | LSD ↓ | MRSTFT ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EnCodec 48 kHz | 3 kbps | 5355 | 0.104829 | 2.92602 | 3.50565 | 5.20039 | 18.3887 | 0.464321 |
| EnCodec 48 kHz | 6 kbps | 5355 | 0.0710060 | 2.00266 | 6.17115 | 7.20794 | 17.9311 | 0.396076 |
| EnCodec 48 kHz | 12 kbps | 5355 | 0.0624319 | 1.40697 | 8.72509 | 9.33567 | 17.6539 | 0.334432 |
| EnCodec 48 kHz | 24 kbps | 5355 | 0.0408916 | 0.981195 | 10.8863 | 11.2612 | 16.7002 | 0.282905 |

### Super-Resolution

| 模型 | 任务 | 样本数 | FAD-CLAP ↓ | FAD-VGGish ↓ | ViSQOL ↑ | SI-SDR ↑ | SNR ↑ | LSD ↓ | LSD-LF ↓ | LSD-HF ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AudioSR basic | 16 kHz -> 48 kHz | 100 | 0.129969 | 2.16932 | 2.27283 | 11.2334 | 5.03170 | 31.0411 | 11.6072 | 37.6990 |
| AudioSR basic | 24 kHz -> 48 kHz | 100 | 0.119498 | 1.93572 | 2.21579 | 11.4233 | 5.31987 | 31.4147 | 21.7898 | 39.5687 |
| AudioSR basic | 32 kHz -> 48 kHz | 100 | 0.118645 | 1.92525 | 2.22748 | 11.4426 | 5.30409 | 31.3664 | 27.4083 | 39.1560 |
| FlowHigh | 16 kHz -> 48 kHz | 100 | 0.117118 | 1.46802 | 1.92407 | 14.8537 | 5.33496 | 43.2012 | 9.01970 | 53.5942 |
| FlowHigh | 24 kHz -> 48 kHz | 100 | 0.110462 | 1.38458 | 2.07041 | 15.4921 | 5.30009 | 42.4387 | 22.0239 | 57.8490 |
| FlowHigh | 32 kHz -> 48 kHz | 100 | 0.110235 | 1.32705 | 2.03987 | 15.4000 | 5.43894 | 42.7335 | 31.1613 | 63.8681 |
| NU-Wave2 | 16 kHz -> 48 kHz | 100 | 0.127181 | 1.42962 | 1.97510 | 16.6728 | 5.38289 | 40.9353 | 8.37480 | 50.7939 |
| NU-Wave2 | 24 kHz -> 48 kHz | 100 | 0.0978331 | 1.29455 | 2.34586 | 18.1820 | 5.63233 | 37.6606 | 15.3709 | 52.9205 |
| NU-Wave2 | 32 kHz -> 48 kHz | 100 | 0.116969 | 1.29614 | 2.84088 | 18.8158 | 5.71048 | 35.0667 | 21.1938 | 57.1455 |
| AERO official 12-48 ckpt | 12 kHz -> 48 kHz | 100 | 0.128436 | 0.833671 | 1.53094 | 13.9046 | 14.3488 | 43.2403 | 5.83694 | 50.5173 |
| FastWave official checkpoint | 16 kHz -> 48 kHz | 100 | 0.140375 | 1.47083 | 2.11795 | 15.5593 | 5.09919 | 46.9476 | 9.87739 | 56.9659 |
| FastWave official checkpoint | 24 kHz -> 48 kHz | 100 | 0.149014 | 1.43500 | 2.24579 | 16.0179 | 5.15691 | 46.4950 | 20.1839 | 62.3169 |
| FastWave official checkpoint | 32 kHz -> 48 kHz | 100 | 0.193116 | 1.44147 | 2.39414 | 16.2034 | 5.15501 | 45.7684 | 27.5530 | 68.8227 |
| UniverSR audio | 16 kHz -> 48 kHz | 100 | 0.580648 | 15.9728 | 2.20985 | -47.2237 | -2.70181 | 37.2084 | 49.7517 | 27.4981 |
| UniverSR audio | 24 kHz -> 48 kHz | 100 | 0.209277 | 9.75928 | 2.31656 | -45.1582 | -2.71183 | 29.9921 | 36.5556 | 19.1425 |

UniverSR 使用官方 `woongzip1/UniverSR` 与 Hugging Face `woongzip1/universr-audio`，官方协议支持 `8/12/16/24 kHz -> 48 kHz`；因此当前只跑 16/24 kHz 两档，不把 32 kHz 作为官方协议结果。AERO 使用官方 repo 与 Google Drive `12-48/aero-nfft=512-hl=128/checkpoint.th`，repo 只带 `4-16` hydra config，adapter 复用同结构并覆盖 `lr_sr=12000/hr_sr=48000`；当前已完成 100 条 MusicCaps-HF 12 kHz -> 48 kHz 推理和 CLAP/VGGish/ViSQOL/距离指标。FastWave 使用官方 `Nikait/FastWave` repo commit `7569045` 与 README Google Drive checkpoint `checkpoint-epoch140.pth`；300 条推理和 CLAP/VGGish/ViSQOL/距离指标已完成。A2SB 是 44.1 kHz mono bandwidth-extension/inpainting 协议，因此单列 `44.1 kHz BWE`，不和当前 `16/24/32 kHz -> 48 kHz` SR 表混作同协议。当前 UniverSR 输出的高频能量显著偏低，CLAP/VGGish 与 SI-SDR 均较差；需要抽听确认是否是官方模型在 MusicCaps-HF music 镜像上的真实性能，还是 adapter 的 bandwidth-limited 输入策略仍有偏差。FlowHigh/NU-Wave2/AERO/FastWave 在 FAD 和 SI-SDR 上较强，但 LSD-HF 明显更高，说明高频频谱形状误差大；SR 排序需要同时看 FAD、ViSQOL、SI-SDR、LSD-LF/HF 和主观听感。

### Separation

| 模型 | stem | 样本数 | FAD-CLAP ↓ | FAD-VGGish ↓ | SI-SDR ↑ | museval SDR ↑ | museval SIR ↑ | museval SAR ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mossland `ckpt/mossland-codec0613` | vocals | 50 | 0.788819 | 5.00457 | -33.2578 | -3.64439 | 3.83420 | -7.08427 |
| Mossland `ckpt/mossland-codec0613` | accompaniment | 50 | 0.644698 | 2.51262 | -19.1808 | -2.14869 | 9.57009 | -5.84080 |
| HTDemucs two-stem | vocals | 50 | 0.242897 | 1.21972 | -1.05150 | 5.65499 | 14.9506 | 8.70221 |
| HTDemucs two-stem | accompaniment | 50 | 0.0576746 | 0.305250 | 18.5629 | 17.9961 | 26.9176 | 20.2880 |
| HTDemucs-FT two-stem | vocals | 50 | 0.212945 | 0.940712 | -4.05274 | 5.92364 | 15.4301 | 8.98561 |
| HTDemucs-FT two-stem | accompaniment | 50 | 0.0561765 | 0.291066 | 16.7488 | 16.8394 | 26.8219 | 18.4925 |
| ByteSep ResUNet143 | vocals | 50 | 0.285021 | 0.904098 | -5.00561 | 4.68830 | 11.6226 | 5.86783 |
| ByteSep ResUNet143 | accompaniment | 50 | 0.110299 | 0.419900 | 18.4457 | 19.4639 | 27.2273 | 20.9069 |
| Open-Unmix UMXHQ | vocals | 50 | 0.416400 | 2.05307 | -7.36111 | 1.37111 | 4.51446 | 5.59311 |
| Open-Unmix UMXHQ | accompaniment | 50 | 0.161922 | 0.619649 | 14.2566 | 15.3188 | 23.7440 | 17.4288 |

HTDemucs 与 HTDemucs-FT 当前走官方 `demucs` CLI 的 `--two-stems vocals` 路径；HTDemucs-FT 是官方 4-model fine-tuned bag，比 `htdemucs` 慢但通常更强。ByteSep 当前走官方 ByteDance `music_source_separation` ResUNet143 Subbandtime checkpoint；Open-Unmix 当前走官方 `openunmix.utils.load_separator()` 与 `openunmix.predict.separate()`，四 stem 输出时把 drums/bass/other 相加为 accompaniment。四者均为 10 秒工程评测，不是 full-track SiSEC median SDR 复现。

补充 full-track 进展：已生成 `musdb18hq_test_fulltrack_manifest.jsonl`，Mossland chunked full-track、HTDemucs、HTDemucs-FT、Demucs MDX、Open-Unmix 和 ByteSep ResUNet143 的 full-track estimates 与 `museval_eval.py --keep-going` 均已 50/50 tracks 有效完成。旧 Mossland full-track wrapper 只得到 15/50 首有效 track，另外 35 首被 BSS Eval v4 的 all-zero reference source 检查拒绝；后续排查确认这不是 MUSDB reference 或路径问题，而是旧 Mossland full-track predictions 实际全部只有 `1.509297s`，reference tracks 是 `76.19s` 到 `430.38s`。旧 `museval_eval.py` 按 prediction/reference 的最短长度截断后，只把歌曲开头 `1.509s` 送入 `museval`，35 首在这个开头短片段里至少有一个 reference stem 全零。根因是 `MosslandCodecTransformer.generate_waveform()` 当前固定 crop 长度生成。旧 Mossland full-track 结果因此明确无效，不进入 full-track 表。现已新增 `fulltrack_infer.py`，在 eval 层用 chunked full-track inference 拼接整首输出；全量 manifest 的 `metadata.prediction_source_length_ratio` 均为 `1.0`，paired-source `museval_eval.py --keep-going` 无 errors。`museval_eval.py` 现已新增 `--length-check auto|always|off`，默认 `auto` 会保护 full-track rows，避免明显短预测被静默截断成误导性 full-track 指标。

#### MUSDB18-HQ full-track museval diagnostics

| 模型 | stem | 有效 tracks | museval SDR mean ↑ | museval SDR median ↑ | frame-median SDR mean ↑ | frame-median SDR median ↑ | museval SIR mean ↑ | museval SAR mean ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mossland chunked full-track | vocals | 50 | -3.47622 | -2.99503 | -1.95546 | -1.92514 | 0.993996 | -23.0928 |
| Mossland chunked full-track | accompaniment | 50 | -2.47170 | -2.41407 | -2.43094 | -2.37595 | 14.2376 | -24.2739 |
| HTDemucs two-stem | vocals | 50 | 6.02866 | 7.05570 | 7.87818 | 8.55366 | 13.2496 | 7.24843 |
| HTDemucs two-stem | accompaniment | 50 | 15.9517 | 16.5595 | 14.8475 | 15.0374 | 25.7060 | 16.1944 |
| HTDemucs-MMI | vocals | 50 | 6.18269 | 7.33625 | 8.04703 | 8.58921 | 13.1316 | 7.40849 |
| HTDemucs-MMI | accompaniment | 50 | 16.1895 | 16.7059 | 15.0057 | 14.8839 | 25.9801 | 16.2311 |
| HTDemucs-FT two-stem | vocals | 50 | 6.36377 | 7.21603 | 8.15088 | 8.59577 | 13.7729 | 7.50154 |
| HTDemucs-FT two-stem | accompaniment | 50 | 14.4082 | 14.8136 | 13.8433 | 14.2095 | 25.8751 | 14.8003 |
| Demucs MDX | vocals | 50 | 5.40078 | 6.20713 | 7.31617 | 8.05874 | 12.1684 | 6.67584 |
| Demucs MDX | accompaniment | 50 | 13.4701 | 13.3385 | 12.9221 | 13.2730 | 23.9565 | 14.1582 |
| Demucs MDX Extra | vocals | 50 | 7.23599 | 7.48279 | 8.85684 | 9.32424 | 14.2718 | 8.18603 |
| Demucs MDX Extra | accompaniment | 50 | 17.7316 | 17.9053 | 16.0339 | 15.8054 | 26.7106 | 17.4354 |
| Open-Unmix UMXHQ | vocals | 50 | 2.76104 | 3.41831 | 5.30036 | 6.37992 | 7.54761 | 5.56736 |
| Open-Unmix UMXHQ | accompaniment | 50 | 14.0779 | 14.1666 | 12.2184 | 12.7769 | 22.3296 | 14.3176 |
| Open-Unmix UMXL | vocals | 50 | 5.00559 | 5.93186 | 6.76080 | 7.36670 | 10.1842 | 6.35993 |
| Open-Unmix UMXL | accompaniment | 50 | 15.2555 | 15.2874 | 13.3633 | 13.4895 | 23.5543 | 15.3721 |
| BS-RoFormer v1.0.12 | vocals | 50 | 9.44030 | 10.0046 | 10.6131 | 10.9930 | 17.6781 | 9.58513 |
| BS-RoFormer v1.0.12 | accompaniment | 50 | 18.2987 | 18.3406 | 17.3349 | 17.3488 | 30.4478 | 18.1904 |
| SCNet XL IHF v1.0.15 | vocals | 50 | 8.67012 | 9.80024 | 10.8688 | 11.5581 | 17.6062 | 9.75794 |
| SCNet XL IHF v1.0.15 | accompaniment | 50 | 20.1550 | 20.4397 | 18.3713 | 17.6201 | 29.1223 | 19.6959 |
| ByteSep ResUNet143 | vocals | 50 | 5.96973 | 7.00048 | 7.08617 | 8.42693 | 12.9031 | 6.81297 |
| ByteSep ResUNet143 | accompaniment | 50 | 17.2533 | 17.3685 | 14.8678 | 14.5157 | 24.5444 | 16.3685 |

#### MUSDB18-HQ full-track distance diagnostics

本表来自同一 full-track prediction manifest 的 `run.py --fad-backend none`，用于补充 waveform/STFT 距离指标；不是 SiSEC 官方排序指标。

| 模型 | stem | 样本数 | SI-SDR dB ↑ | SNR dB ↑ | LSD ↓ | MRSTFT ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mossland chunked full-track | vocals | 50 | -50.8738 | -1.91692 | 49.2881 | 1.02326 |
| Mossland chunked full-track | accompaniment | 50 | -44.1226 | -2.44062 | 24.1023 | 0.804338 |
| HTDemucs two-stem | vocals | 50 | 7.85745 | 8.56662 | 42.8458 | 0.362364 |
| HTDemucs two-stem | accompaniment | 50 | 14.6349 | 14.6318 | 11.7817 | 0.148055 |
| HTDemucs-MMI | vocals | 50 | 8.12349 | 8.87042 | 42.5763 | 0.343884 |
| HTDemucs-MMI | accompaniment | 50 | 14.7610 | 14.8573 | 12.7317 | 0.137832 |
| HTDemucs-FT two-stem | vocals | 50 | 8.16641 | 8.81731 | 40.8341 | 0.355166 |
| HTDemucs-FT two-stem | accompaniment | 50 | 13.7107 | 13.6980 | 12.2766 | 0.163972 |
| Demucs MDX | vocals | 50 | 6.97355 | 8.11098 | 47.3609 | 0.391326 |
| Demucs MDX | accompaniment | 50 | 12.8190 | 12.9669 | 14.6468 | 0.174720 |
| Demucs MDX Extra | vocals | 50 | 9.36715 | 9.89831 | 43.7666 | 0.314573 |
| Demucs MDX Extra | accompaniment | 50 | 16.1144 | 16.1446 | 13.5941 | 0.118835 |
| Open-Unmix UMXHQ | vocals | 50 | 4.42597 | 6.09432 | 48.1082 | 0.631494 |
| Open-Unmix UMXHQ | accompaniment | 50 | 12.0328 | 12.3121 | 20.3290 | 0.194696 |
| Open-Unmix UMXL | vocals | 50 | 5.78740 | 7.70327 | 55.2420 | 0.435824 |
| Open-Unmix UMXL | accompaniment | 50 | 13.3919 | 13.5979 | 44.8733 | 0.181986 |
| BS-RoFormer v1.0.12 | vocals | 50 | 10.3358 | 11.0079 | 34.8377 | 0.296649 |
| BS-RoFormer v1.0.12 | accompaniment | 50 | 16.8025 | 16.8874 | 10.8076 | 0.108195 |
| SCNet XL IHF v1.0.15 | vocals | 50 | 10.6988 | 11.4220 | 48.7232 | 0.309141 |
| SCNet XL IHF v1.0.15 | accompaniment | 50 | 17.6290 | 17.6980 | 15.0125 | 0.102404 |
| ByteSep ResUNet143 | vocals | 50 | 7.12089 | 8.12068 | 36.9941 | 0.469286 |
| ByteSep ResUNet143 | accompaniment | 50 | 14.4050 | 14.5921 | 10.1824 | 0.140230 |

#### MUSDB18-HQ full-track FAD-CLAP diagnostics

本表来自同一 full-track prediction manifest 的 `run.py --fad-backend clap`，用于补充分布级 CLAP embedding 距离；不是 SiSEC 官方排序指标。

| 模型 | stem | 样本数 | FAD-CLAP ↓ | SI-SDR dB ↑ | SNR dB ↑ | LSD ↓ | MRSTFT ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mossland chunked full-track | vocals | 50 | 0.463199 | -50.8738 | -1.91692 | 49.2881 | 1.02326 |
| Mossland chunked full-track | accompaniment | 50 | 0.252407 | -44.1226 | -2.44062 | 24.1023 | 0.804338 |
| HTDemucs two-stem | vocals | 50 | 0.292923 | 7.85745 | 8.56662 | 42.8458 | 0.362364 |
| HTDemucs two-stem | accompaniment | 50 | 0.159056 | 14.6349 | 14.6318 | 11.7817 | 0.148055 |
| HTDemucs-MMI | vocals | 50 | 0.311829 | 8.12349 | 8.87042 | 42.5763 | 0.343884 |
| HTDemucs-MMI | accompaniment | 50 | 0.176019 | 14.7610 | 14.8573 | 12.7317 | 0.137832 |
| HTDemucs-FT two-stem | vocals | 50 | 0.249566 | 8.16641 | 8.81731 | 40.8341 | 0.355166 |
| HTDemucs-FT two-stem | accompaniment | 50 | 0.110911 | 13.7107 | 13.6980 | 12.2766 | 0.163972 |
| Demucs MDX | vocals | 50 | 0.304748 | 6.97355 | 8.11098 | 47.3609 | 0.391326 |
| Demucs MDX | accompaniment | 50 | 0.154081 | 12.8190 | 12.9669 | 14.6468 | 0.174720 |
| Demucs MDX Extra | vocals | 50 | 0.341449 | 9.36715 | 9.89831 | 43.7666 | 0.314573 |
| Demucs MDX Extra | accompaniment | 50 | 0.137790 | 16.1144 | 16.1446 | 13.5941 | 0.118835 |
| Open-Unmix UMXHQ | vocals | 50 | 0.448585 | 4.42597 | 6.09432 | 48.1082 | 0.631494 |
| Open-Unmix UMXHQ | accompaniment | 50 | 0.173807 | 12.0328 | 12.3121 | 20.3290 | 0.194696 |
| Open-Unmix UMXL | vocals | 50 | 0.435731 | 5.78740 | 7.70327 | 55.2420 | 0.435824 |
| Open-Unmix UMXL | accompaniment | 50 | 0.137541 | 13.3919 | 13.5979 | 44.8733 | 0.181986 |
| BS-RoFormer v1.0.12 | vocals | 50 | 0.308159 | 10.3358 | 11.0079 | 34.8377 | 0.296649 |
| BS-RoFormer v1.0.12 | accompaniment | 50 | 0.162335 | 16.8025 | 16.8874 | 10.8076 | 0.108195 |
| SCNet XL IHF v1.0.15 | vocals | 50 | 0.375005 | 10.6988 | 11.4220 | 48.7232 | 0.309141 |
| SCNet XL IHF v1.0.15 | accompaniment | 50 | 0.128831 | 17.6290 | 17.6980 | 15.0125 | 0.102404 |
| ByteSep ResUNet143 | vocals | 50 | 0.273098 | 7.12089 | 8.12068 | 36.9941 | 0.469286 |
| ByteSep ResUNet143 | accompaniment | 50 | 0.151869 | 14.4050 | 14.5921 | 10.1824 | 0.140230 |

## 当前实现状态

| 范围 | 状态 | 证据 |
| --- | --- | --- |
| 固定 manifest | 已实现 | `build_manifest.py` 可从 prepared `mixture/vocals/accompaniment` 目录生成五任务 JSONL。 |
| 单音频多任务 manifest | 已实现 | `build_audio_task_manifest.py` 可从已有 reconstruction/audio manifest 派生 reconstruction、super-resolution bucket 和 mono-to-stereo 多 seed 样本。 |
| Mossland checkpoint 推理 | 已 smoke | `ckpt/mossland-codec0613` 可通过 `scripts.factory.load_model(ckpt_dir=...)` 加载，并在 GPU 上跑通 1 秒五任务样本。 |
| 距离/频谱指标 | 已实现 | `SNR`、`SI-SDR`、`LSD`、`MRSTFT`、SR 的 `LSD-LF/LSD-HF/HF energy`、stereo 的 fold-down/width/correlation。 |
| FAD proxy | 已实现 | `fad_mel_proxy` 可用于流程测试，但不是论文官方 FAD。 |
| CoDiCodec FAD_clap | 已实现并 smoke | `--fad-backend clap` 使用 Hugging Face `laion/clap-htsat-unfused`；绕过 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 后已下载到 `checkpoints/clap`，随机音频 512 维 embedding smoke 通过。 |
| CoDiCodec VGGish FAD | 已实现并 smoke | `--fad-backend vggish` 使用 `tmp/eval_metric_refs/torchvggish` 的 PyTorch VGGish port、raw 128-D embedding 和权重 `vggish-10086976.pth`；10 条 MusicCaps-HF smoke 已输出 `fad_vggish`。该路径避开 TF1/Beam，但正式报告需标注后端为 `torchvggish_raw`。 |
| ViSQOL | 已实现并跑通 | `visqol_eval.py` 调 Google ViSQOL binary，把 reference/prediction 转成 48 kHz mono 16-bit WAV，使用 audio-mode `libsvm_nu_svr_model.txt` 输出 MOS-LQO；已完成 Mossland 100-source multitask ViSQOL。 |
| museval/BSS Eval v4 | 已实现 10 秒工程评测 | `museval_eval.py` 通过局部 `imp.reload` shim 绕过 Python 3.12/future 兼容问题，直接调用 `museval.evaluate(..., mode="v4")`。MUSDB18-HQ test 起点 0 的 50 首 10 秒片段中 15 首有效、35 首因某个 reference source 全静音被 BSS Eval v4 拒绝。 |
| Mossland chunked full-track inference | 已完成 full-track 评测 | `fulltrack_infer.py` 对 full-track separation rows 按模型固定 `66560` frame 窗口做 chunked `generate_waveform()`，默认 `0.5s` overlap 并拼接回整首长度。全量 MUSDB18-HQ test 100 行 manifest 已完成，`metadata.prediction_source_length_ratio=1.0`，`museval_eval.py --keep-going` 50/50 tracks 有效且无 errors。 |
| mir_eval BSS diagnostic | 已实现 | separation 任务输出 `mir_eval_bss_images_sdr_db/isr_db/sar_db`，用于单源 teacher-stem 诊断；现在同时有 `museval_eval.py` 的 paired-source v4 工程评测。 |
| MusicCaps metadata | 已下载 | `scripts/mossland-codec/eval_benchmark/data/musiccaps/musiccaps_train_metadata.jsonl`，5521 条。 |
| MusicCaps audio | 未下载 | 直连 YouTube `Network is unreachable`；走环境代理可访问但 YouTube 要求 bot 验证，需要 `yt-dlp --cookies` 或内部音频镜像。`download_musiccaps_audio.py` 已支持 `--use-env-proxy`、`--cookies`、`--cookies-from-browser` 和超时。 |
| MusicCaps HF 镜像 | 已下载 | 第三方 `mahendra0203/musiccaps_processed_full` 已下载 5355 个 wav，`hf_musiccaps_reconstruct_manifest.jsonl` 有 5355 行，缺失 166 条；不是 Google 官方音频源，正式表格必须标注镜像来源并确认许可。 |
| MUSDB18-HQ | 已下载并解压 | Zenodo zip 已完成并校验大小 `22656664047` 字节；`test/` split 有 50 首 wav stem，已生成 100 行 10 秒 separation manifest，并跑完 Mossland 100 条 separation eval。 |
| MUSDB18 | 已下载 | compressed MUSDB18 zip 已完成并校验大小 `4684228845` 字节；作为 fallback，当前 manifest builder 优先支持 MUSDB18-HQ wav stem 结构。 |
| 长跑恢复 / 分片 | 已实现 | `run.py` 默认复用确定性输出路径下已有的非空 prediction wav，避免中断后重复占用 GPU；传 `--overwrite-predictions` 可强制重算；新增 `--max-items`、`--num-shards/--shard-id` 和 `--progress-every`，方便 long-form CLAP/VGGish/FAD 任务拆分到多个独立输出目录并观察进度；`aggregate_results.py` 可合并 shard `results.jsonl` 并重新聚合全量 FAD/距离 summary。 |
| 公开数据下载器 | 已增强 | `download_public_dataset.py` 支持 `--no-proxy`、`--head`、`.part` 断点、registry 文件大小校验和周期性进度输出。 |
| EnCodec baseline | full 推理和指标已完成 | `baselines/encodec_baseline.py` 使用官方 `encodec==0.1.1`、`EncodecModel.encodec_model_48khz()` 和 torch cache 中的 `encodec_48khz-7e698e3e.th`；10 条 MusicCaps-HF smoke 已评测，5355 条 full run 已写 `encodec_manifest.jsonl` 并完成 CLAP/VGGish 指标。 |
| DAC baseline | full 推理和指标已完成 | `baselines/dac_baseline.py` 使用官方 `descript-audio-codec`、44kHz 8kbps checkpoint `tmp/eval_baseline_refs/dac/checkpoints/weights.pth`；10 条 MusicCaps-HF smoke 已评测。5355 条 full run 已从单路改为 4 个 disjoint shard 并行跑完，`runs/baselines_musiccaps_full/dac_manifest.jsonl` 已合并 5355 行；full CLAP/VGGish 指标已输出。 |
| AudioSR baseline | 已接入并完成 100 条/档工程评测 | `baselines/sr_baseline.py` 使用官方 AudioSR 仓库和 Hugging Face `haoheliu/audiosr_basic` checkpoint；MusicCaps-HF SR@16000/24000/32000 各 100 条已用默认 50 DDIM steps/guidance 3.5 跑完，并完成 CLAP/VGGish/ViSQOL/距离指标。 |
| DiffStereo baseline | 已接入并完成 20 条 official sampler 小评测 | `baselines/diffstereo_baseline.py` 使用官方 DiffStereo 仓库和真实 checkpoint `tmp/eval_baseline_refs/diffstereo/DiffStereo/checkpoints/model_epoch_80000.pt`；MusicCaps-HF mono-to-stereo seed0 前 20 条已用空 `timestep_respacing` 的 1000-step sampler 跑完，并完成 CLAP/VGGish/ViSQOL/空间指标。注意 `tmp/eval_baseline_refs/DiffStereo/...` 大写路径里的同名 checkpoint 仍是 Git LFS pointer，不能用。 |
| ViSQOL / museval v4 | 已接入 | ViSQOL 已用 Bazel 5.1.0 本地构建并通过 wrapper 跑通；`museval_eval.py` 用局部 shim 解决 Python 3.12 的 `imp` 兼容问题。 |

## 官方 baseline 协议核对

| Baseline / protocol | 当前口径 | 复现状态 |
| --- | --- | --- |
| EnCodec | 使用官方 Facebook Research `encodec_model_48khz()`、48 kHz stereo、24 kbps；adapter 额外 pad 到官方 segment grid 后再裁回原长，避免尾段 overlap-add shape mismatch。 | 对 48 kHz music / 24 kbps baseline 基本一致；注意官方 CLI 默认 bandwidth 是 6 kbps，报告中需写明当前 bandwidth。 |
| DAC | 使用官方 `descript-audio-codec` 44.1 kHz 8 kbps 权重，走官方 `compress()` / `decompress()`，默认 `normalize_db=-16` 和 5 秒 window。 | 与官方推理路径一致；如果要比较无响度处理误差，需要另跑消融，不能混作默认官方结果。 |
| AudioSR | 接官方 `haoheliu/audiosr_basic` checkpoint，默认 50 DDIM steps、guidance 3.5。 | 已完成 MusicCaps-HF SR@16000/24000/32000 各 100 条工程评测；不是 AudioSR 论文级复现，因为数据集、主观 MOS/MUSHRA 和完整官方 benchmark 仍未对齐。 |
| FlowHigh | 接官方/维护仓库 `resemble-ai/flowhigh` 与本地 checkpoint，当前使用 48 kHz target、`time_step=1` 的工程推理路径。 | 已完成 MusicCaps-HF SR@16000/24000/32000 各 100 条工程评测；不是论文级复现，因为官方 benchmark 数据和主观评测未对齐。 |
| NU-Wave2 | 接官方仓库与官方 48 kHz target checkpoint，适配多输入采样率到 48 kHz 的 SR 任务。 | 已完成 MusicCaps-HF SR@16000/24000/32000 各 100 条工程评测；不是 NU-Wave2 论文级复现，因为当前数据是 MusicCaps-HF music 镜像而非论文 speech benchmark。 |
| UniverSR | 接官方 `woongzip1/UniverSR` 与 HF `woongzip1/universr-audio` checkpoint，官方支持 `8/12/16/24 kHz -> 48 kHz`。当前 adapter 使用 48 kHz reference 先 downsample/up 到有效低采样率，再调用官方 `UniverSR.enhance(..., input_sr=...)`。 | 已完成 MusicCaps-HF SR@16000/24000 各 100 条推理和 CLAP/VGGish/ViSQOL/距离指标。不是论文级复现，因为数据集不是 UniverSR paper benchmark，且 32 kHz 不属于官方支持输入采样率。 |
| DiffStereo | adapter 默认官方 `model_epoch_80000.pt`、24 kHz、9.1 秒 chunk、空 `timestep_respacing` 的 1000-step diffusion。 | 已完成 MusicCaps-HF mono-to-stereo seed0 前 20 条 official sampler 工程评测；不是 DiffStereo 论文级复现，因为官方 demo/benchmark 数据和主观 MOS 尚未对齐。 |
| s3a | 官方 repo `s3a-spatialaudio/s3a-decorrelation-toolbox` 已 clone，当前 adapter 默认优先官方 API，失败时回退到 mono-compatible all-pass decorrelator。 | 已完成 MusicCaps-HF mono-to-stereo seed0/1 前 20 条 fallback DSP 工程评测；这是 DSP anchor，不是神经生成式 stereo SOTA。 |
| Ambisonizer | 官方 repo `yongyizang/ambisonizer` 已 clone 到 `tmp/eval_baseline_refs/ambisonizer/ambisonizer`，adapter 使用官方 `SEANet(480000, 64)`，44.1 kHz 固定长度输入，预测 FOA `X/Y`，再用 mono 条件作为 `W` 并按官方 `synthesize.ipynb` 方位 `[-60, 120]` 解码 stereo。 | 官方 Google Drive checkpoint 已下载到 `checkpoints/ambisonizer/ambisonizer_state_dict.pt`，SHA256 `1e40b98a21a464c77081cdf2c96a2a8944773a29c1964c77489ba607a754ce69`；1 条 smoke 和 MusicCaps-HF seed0 100 条工程评测已完成。 |
| SNAC | 使用官方 `hubertsiuzdak/snac_44khz`，当前 adapter 对 MusicCaps-HF 重采样到 44.1 kHz 并按官方模型推理。 | 已完成 full 5355 条 CLAP/VGGish/距离工程评测；SNAC 为 mono codec，和 Mossland stereo codec 比较时需声明声道策略。 |
| WavTokenizer | 使用官方 WavTokenizer music 75-token checkpoint，当前 adapter 按官方 24 kHz mono tokenizer 路径推理。 | 已完成 full 5355 条 CLAP/VGGish/距离工程评测；需要声明重采样/mono 策略，不能视作 stereo codec 论文复现。 |
| CoDiCodec | 使用官方 `SonyCSLParis/codicodec` 推理代码和 checkpoint，当前跑 MusicCaps-HF reconstruction。 | 已完成 full 5355 条 CLAP/VGGish/距离工程评测；官方仓库未提供评估脚本，因此当前指标是本仓库统一 pipeline 工程评测。 |
| CoDiCodec FAD | 当前 `fad_clap` 使用 HF `laion/clap-htsat-unfused`；`fad_vggish` 使用 PyTorch `torchvggish_raw`，每条音频 frame embedding 先均值再汇总 Frechet distance。 | 可运行的是 CoDiCodec-style 后端，不是官方脚本复现；官方仓库未提供评估脚本，VGGish/CLAP 模型与聚合方式必须在表格旁说明。 |
| HTDemucs | 使用官方 `demucs` CLI 的 `--two-stems vocals` 推理路径，当前评 vocals 与 accompaniment；full-track 表同时记录 `htdemucs`、`htdemucs_ft` 和 `htdemucs_mmi`。 | 已完成 MUSDB18-HQ test 非静音 10 秒工程评测，以及多种 Demucs 官方/公开权重 full-track 工程评测；`htdemucs_mmi` 属于额外训练/变体口径，不能直接混作公平 MUSDB-only reproduction。 |
| ByteSep | 使用官方 ByteDance `music_source_separation` repo、Zenodo ResUNet143 Subbandtime vocals/accompaniment checkpoints 和 MUSDB18 config；adapter 通过轻量 shim 避免安装旧 Lightning 依赖。 | 已完成 MUSDB18-HQ test 非静音 10 秒工程评测；不是论文/full-track 复现。MobileNet 小权重也已 smoke。 |
| ZFTurbo/MVSep MSST | 新增 `msst_baseline.py`，目标接 `ZFTurbo/Music-Source-Separation-Training` 的 BS-RoFormer / Mel-Band RoFormer / SCNet 等模型，调用官方 `inference.py` 并按本仓库 manifest 输出 vocals/accompaniment predictions。 | Adapter CLI/help 和轻量单元测试已通过；BS-RoFormer MUSDB18HQ-only v1.0.12 config/checkpoint 已完成 full-track 推理、museval 和 distance 评测，并已启动 CLAP 评测。SCNet XL IHF v1.0.15 config/checkpoint 已完成 full-track 推理，museval/distance 后评测正在跑。后续需记录具体 config、checkpoint、commit、license 和是否使用额外训练数据，不能直接混作公平 MUSDB paper reproduction。 |
| Demucs MDX Extra | 使用官方 `demucs` CLI 的 `mdx_extra` bag 权重和 `--two-stems vocals` 路径。 | 已完成 MUSDB18-HQ full-track museval/distance 工程评测；这是 extra-data 变体，存在训练数据与 MUSDB test 口径污染风险，只作为上界/工程 baseline，不混入公平 MUSDB-only reproduction。 |
| Open-Unmix | 使用官方 `openunmix.utils.load_separator()` 与 `openunmix.predict.separate()`，四 stem 输出时把 drums/bass/other 相加为 accompaniment；full-track 表同时记录 UMXHQ 和 UMXL。 | UMXHQ 已完成 MUSDB18-HQ test 非静音 10 秒和 full-track 工程评测；UMXL 已完成 full-track museval/distance 评测，需标注为额外/非标准训练数据口径，不混作公平 MUSDB-only reproduction。 |
| MUSDB / museval | 已有 MUSDB18-HQ test 非静音 10 秒 manifest、`mir_eval` diagnostic 和 paired-source `museval` v4 工程评测。 | 仍不是 SiSEC full-track `museval` v4；当前非静音 10 秒窗口避免了 all-zero source 错误，但不能直接对比 full-track median SDR。 |
| ViSQOL | 使用 Google ViSQOL 本地 Bazel binary、audio mode、48 kHz mono 16-bit WAV、`libsvm_nu_svr_model.txt`。 | 已可运行；stereo/upmix 任务会先 downmix 到 mono，因此只能作为音频质量 MOS-LQO，不能评价空间感。 |
| AERO | 使用官方 repo `tmp/eval_baseline_refs/aero` 与 Google Drive `12-48/aero-nfft=512-hl=128/checkpoint.th`；官方 repo 只带 `4-16` hydra config，adapter 复用同模型结构并覆盖 `lr_sr=12000/hr_sr=48000`。 | 已完成 MusicCaps-HF 12 kHz -> 48 kHz 100 条工程评测；不是 AERO paper reproduction，因为当前数据不是 AERO speech/music 官方 benchmark，且 adapter 只覆盖 12-48 checkpoint。 |
| FastWave | 使用官方 `Nikait/FastWave` repo commit `7569045` 和 README Google Drive checkpoint `checkpoint-epoch140.pth`；协议是 general audio SR `any -> 48 kHz`，当前 adapter 按 manifest `low_sample_rate` 生成 Chebyshev low-pass/down-up 条件，默认 8 EDM steps。 | MusicCaps-HF 16/24/32 kHz 三个 SR bucket 各 100 条推理和 CLAP/VGGish/ViSQOL/距离指标均已完成。论文主评测偏 VCTK speech，因此当前结果标为 MusicCaps-HF 工程评测，不是 FastWave paper reproduction。 |
| AEROMamba | 官方 repo `aeromamba-super-resolution/aeromamba` commit `67d3c8a`，官方协议 `11.025 kHz -> 44.1 kHz`，README checkpoint 在 SharePoint，当前无法匿名下载；本地 smoke 使用 HF 镜像 `innova-ai/AEROMamba` 的 `checkpoint.th`。 | 1 条官方 `predict.py` smoke 已跑通，但当前 Python 3.12/Torch 2.7 环境无法正常安装 `mamba-ssm`，worker 用临时 shim 绕过 Mamba extension 导入，且 forward 路径未实际调用 Mamba CUDA kernel。严格复现需要 Python 3.10/Torch 1.12.1/CUDA 11.3 隔离环境；暂不放入主表。 |
| NU-Wave | 本地有官方仓库 `tmp/eval_baseline_refs/nuwave`，但缺官方 checkpoint；默认协议是固定 x2 `24->48 kHz`。 | 未复现。可作为 x2 SR baseline，但不适合当前多 bucket MusicCaps-HF SR 表，除非单独定义 `24->48 kHz` 协议。 |

## CoDiCodec 口径

`docs/papers/codicodec.pdf` 在 MusicCaps 上报告 `SI-SDR`、`ViSQOL`、VGGish `FAD` 和 CLAP `FAD_clap`。本地已拉取 `tmp/codicodec`，但官方仓库当前只提供 codec 推理代码，没有评估脚本。因此后续复现 CoDiCodec 表格需要自行固定：

- MusicCaps evaluation manifest。
- VGGish checkpoint 与特征抽取实现。
- CLAP checkpoint，当前计划使用 `laion/clap-htsat-unfused`。
- continuous/discrete、autoregressive/parallel decoding 设置。

## Mossland Smoke 结果

命令：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.run \
  --manifest scripts/mossland-codec/eval_benchmark/data/tmp_34078087_manifest.jsonl \
  --checkpoint-dir ckpt/mossland-codec0613 \
  --device cuda \
  --metrics-device cuda \
  --fad-backend none \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/smoke_tmp_34078087
```

结果文件：

- `scripts/mossland-codec/eval_benchmark/runs/smoke_tmp_34078087/results.jsonl`
- `scripts/mossland-codec/eval_benchmark/runs/smoke_tmp_34078087/summary.json`

| 任务 | 样本数 | SI-SDR dB | SNR dB | LSD | MRSTFT | 额外指标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| reconstruct | 1 | -22.35 | -2.17 | 16.54 | 0.568 | - |
| separate_vocals | 1 | -37.37 | -2.25 | 33.05 | 0.811 | teacher vocal reference |
| separate_accompaniment | 1 | -32.84 | -2.93 | 16.48 | 0.628 | teacher accompaniment reference |
| super_resolution@16000 | 1 | -19.64 | -1.96 | 17.09 | 0.573 | `LSD-LF=8.95`、`LSD-HF=20.31`、`HF energy=0.00965` |
| mono_to_stereo | 1 | -22.19 | -2.22 | 17.05 | 0.574 | fold-down SI-SDR `-21.76`、width `0.400`、corr `0.724` |

这个 smoke 只证明 pipeline 与 checkpoint 连通，不代表模型真实质量；片段只有 1 秒且 FAD 关闭。

## MusicCaps-HF 10 条 Smoke 对比

以下结果使用第三方 MusicCaps HF 镜像的前 10 条 reconstruction clip。`FAD_clap` 使用 `--fad-backend clap`、`--metrics-device cuda`；`FAD_vggish` 使用 `--fad-backend vggish`、`--metrics-device cpu`。这张表只用于 pipeline/baseline smoke，不是官方 MusicCaps 音频的论文复现。

| 模型 | 样本数 | FAD_clap | FAD_vggish | SI-SDR dB | SNR dB | LSD | MRSTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mossland `ckpt/mossland-codec0613` | 10 | 0.662314 | 7.99991 | -9.19217 | -0.0139046 | 27.4147 | 0.493434 |
| EnCodec 48kHz 24kbps | 10 | 0.118668 | 4.92134 | 12.4967 | 11.7355 | 17.0104 | 0.264840 |
| DAC 44kHz 8kbps | 10 | 0.176544 | 2.40878 | 11.4817 | 5.44491 | 23.1064 | 0.525454 |

## MusicCaps-HF Full Reconstruction

以下 full 结果使用第三方 MusicCaps-HF 镜像 5355 条 reconstruction clip，不是 Google 官方音频发布。

| 模型 | 样本数 | FAD_clap | FAD_vggish | SI-SDR dB | SNR dB | LSD | MRSTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EnCodec 48kHz 24kbps | 5355 | 0.0408916 | 0.981195 | 10.8863 | 11.2612 | 16.7002 | 0.282905 |
| DAC 44kHz 8kbps | 5355 | 0.075158 | 0.713645 | 9.72614 | 4.72790 | 22.4124 | 0.557487 |
| SNAC 44kHz | 5355 | 0.0563409 | 1.64879 | 2.13932 | 4.23513 | 17.1182 | 0.462963 |
| WavTokenizer music 75-token | 5355 | 0.0764699 | 1.61131 | -3.28095 | 1.72287 | 18.1491 | 0.541830 |
| CoDiCodec official checkpoint | 5355 | 0.0915413 | 0.496719 | -1.12868 | 2.37269 | 20.8388 | 0.490066 |

## MusicCaps-HF 10 条 Mossland 多任务评测

`build_audio_task_manifest.py` 从 `hf_musiccaps_reconstruct_manifest.jsonl` 前 10 条派生 60 行：10 条 reconstruction、3 个 SR bucket 各 10 条、2 个 stereo seed 共 20 条。结果来自 `ckpt/mossland-codec0613`，`--device cuda:1`、`--metrics-device cuda:1`、`--fad-backend vggish`，输出目录为 `scripts/mossland-codec/eval_benchmark/runs/mossland_musiccaps_multitask10_vggish/`。该表仍使用第三方 MusicCaps-HF 音频镜像，只是工程评测。

| 任务 | 样本数 | FAD_vggish | SI-SDR dB | SNR dB | LSD | MRSTFT | 额外指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| reconstruct | 10 | 7.99986 | -9.19217 | -0.0139046 | 27.4147 | 0.493434 | - |
| super_resolution@16000 | 10 | 8.25946 | -9.04064 | -0.00387318 | 26.0008 | 0.477643 | `LSD-LF=10.7019`、`LSD-HF=31.5382`、`HF energy=0.000241936` |
| super_resolution@24000 | 10 | 8.26317 | -9.18282 | -0.0295632 | 27.6202 | 0.495991 | `LSD-LF=18.6325`、`LSD-HF=35.3907`、`HF energy=0.000328005` |
| super_resolution@32000 | 10 | 8.09181 | -9.20226 | -0.0260808 | 27.3882 | 0.495059 | `LSD-LF=23.1051`、`LSD-HF=36.0290`、`HF energy=0.0000175585` |
| mono_to_stereo | 20 | 6.77844 | -7.86920 | 0.270134 | 27.1892 | 0.507475 | fold-down SI-SDR `-6.78983`、width `0.116744`、corr `0.962857` |

## MusicCaps-HF 100-source Mossland 多任务评测

`hf_musiccaps_multitask_manifest_100.jsonl` 从第三方 MusicCaps-HF 镜像前 100 条音频派生 600 行：100 条 reconstruction、3 个 SR bucket 各 100 条、2 个 stereo seed 共 200 条。Mossland 推理结果来自 `runs/mossland_musiccaps_multitask100_vggish/results.jsonl`；CLAP 指标复用同一 prediction manifest 重算；ViSQOL 用 `visqol_eval.py` 按 bucket 并发调用 Google ViSQOL binary。该表仍是工程评测，不是 Google 官方 MusicCaps 音频的论文复现。

| 任务 | 样本数 | FAD_clap | FAD_vggish | ViSQOL MOS-LQO | SI-SDR dB | SNR dB | LSD | MRSTFT | 额外指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| reconstruct | 100 | 0.450071 | 3.87111 | 2.93772 | -12.4447 | -0.748526 | 28.2961 | 0.544861 | - |
| super_resolution@16000 | 100 | 0.438918 | 3.81445 | 3.28623 | -12.3951 | -0.732761 | 26.3007 | 0.520891 | `LSD-LF=10.7756`、`LSD-HF=31.8564`、`HF energy=0.000825977` |
| super_resolution@24000 | 100 | 0.449449 | 3.77166 | 2.91487 | -12.3649 | -0.771458 | 28.5929 | 0.547968 | `LSD-LF=19.3639`、`LSD-HF=36.5764`、`HF energy=0.00100527` |
| super_resolution@32000 | 100 | 0.448485 | 3.75972 | 2.91912 | -12.4190 | -0.764738 | 28.4977 | 0.546545 | `LSD-LF=24.9741`、`LSD-HF=35.6416`、`HF energy=0.00000929214` |
| mono_to_stereo | 200 | 0.455147 | 3.48015 | 2.81490 | -12.3538 | -0.635905 | 28.4875 | 0.548601 | fold-down SI-SDR `-11.533`、width `0.165253`、corr `0.920662`; ViSQOL downmixes to mono |

## MusicCaps-HF 100 条 SR Baselines

下面使用 `hf_musiccaps_multitask_manifest_100.jsonl` 中三个 `super_resolution` bucket 各 100 条。AudioSR 使用官方 `haoheliu/audiosr_basic` checkpoint、默认 `ddim_steps=50`、`guidance_scale=3.5`；FlowHigh 使用本地官方/维护 checkpoint 的 48 kHz target 推理路径；NU-Wave2 使用官方 48 kHz target checkpoint。该表仍基于 MusicCaps-HF 第三方镜像，不是各论文原始 benchmark 复现。

| 模型 | 任务 | 样本数 | FAD_clap | FAD_vggish | ViSQOL MOS-LQO | SI-SDR dB | SNR dB | LSD | LSD-LF | LSD-HF | MRSTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AudioSR basic | 16 kHz -> 48 kHz | 100 | 0.129969 | 2.16932 | 2.27283 | 11.2334 | 5.03170 | 31.0411 | 11.6072 | 37.6990 | 0.695244 |
| AudioSR basic | 24 kHz -> 48 kHz | 100 | 0.119498 | 1.93572 | 2.21579 | 11.4233 | 5.31987 | 31.4147 | 21.7898 | 39.5687 | 0.680525 |
| AudioSR basic | 32 kHz -> 48 kHz | 100 | 0.118645 | 1.92525 | 2.22748 | 11.4426 | 5.30409 | 31.3664 | 27.4083 | 39.1560 | 0.680780 |
| FlowHigh | 16 kHz -> 48 kHz | 100 | 0.117118 | 1.46802 | 1.92407 | 14.8537 | 5.33496 | 43.2012 | 9.01970 | 53.5942 | 0.963287 |
| FlowHigh | 24 kHz -> 48 kHz | 100 | 0.110462 | 1.38458 | 2.07041 | 15.4921 | 5.30009 | 42.4387 | 22.0239 | 57.8490 | 0.956508 |
| FlowHigh | 32 kHz -> 48 kHz | 100 | 0.110235 | 1.32705 | 2.03987 | 15.4000 | 5.43894 | 42.7335 | 31.1613 | 63.8681 | 0.960439 |
| NU-Wave2 | 16 kHz -> 48 kHz | 100 | 0.127181 | 1.42962 | 1.97510 | 16.6728 | 5.38289 | 40.9353 | 8.37480 | 50.7939 | 0.943361 |
| NU-Wave2 | 24 kHz -> 48 kHz | 100 | 0.0978331 | 1.29455 | 2.34586 | 18.1820 | 5.63233 | 37.6606 | 15.3709 | 52.9205 | 0.922839 |
| NU-Wave2 | 32 kHz -> 48 kHz | 100 | 0.116969 | 1.29614 | 2.84088 | 18.8158 | 5.71048 | 35.0667 | 21.1938 | 57.1455 | 0.910062 |
| AERO official 12-48 ckpt | 12 kHz -> 48 kHz | 100 | 0.128436 | 0.833671 | 1.53094 | 13.9046 | 14.3488 | 43.2403 | 5.83694 | 50.5173 | 0.282232 |
| FastWave official checkpoint | 16 kHz -> 48 kHz | 100 | 0.140375 | 1.47083 | 2.11795 | 15.5593 | 5.09919 | 46.9476 | 9.87739 | 56.9659 | 0.973030 |
| FastWave official checkpoint | 24 kHz -> 48 kHz | 100 | 0.149014 | 1.43500 | 2.24579 | 16.0179 | 5.15691 | 46.4950 | 20.1839 | 62.3169 | 0.967911 |
| FastWave official checkpoint | 32 kHz -> 48 kHz | 100 | 0.193116 | 1.44147 | 2.39414 | 16.2034 | 5.15501 | 45.7684 | 27.5530 | 68.8227 | 0.964851 |
| UniverSR audio | 16 kHz -> 48 kHz | 100 | 0.580648 | 15.9728 | 2.20985 | -47.2237 | -2.70181 | 37.2084 | 49.7517 | 27.4981 | 1.05641 |
| UniverSR audio | 24 kHz -> 48 kHz | 100 | 0.209277 | 9.75928 | 2.31656 | -45.1582 | -2.71183 | 29.9921 | 36.5556 | 19.1425 | 1.01077 |

UniverSR 只列官方支持的 16/24 kHz 输入；100 条 ViSQOL 已完成。AERO 当前按官方 12-48 speech protocol 单列，不与 16/24/32 kHz buckets 视作同一协议。FastWave 三档 100 条 CLAP/VGGish/ViSQOL/距离指标均已完成。当前 UniverSR 的 objective 结果明显异常偏弱，尤其 SI-SDR 与 FAD-VGGish；后续需要抽听并核对官方输入退化策略。FlowHigh/NU-Wave2/AERO/FastWave 在 FAD 和 SI-SDR 上较强，但 LSD-HF 明显更高，说明高频频谱形状误差大；SR 排序需要同时看 FAD、ViSQOL、SI-SDR、LSD-LF/HF 和主观听感。

### MusicCaps-HF 300 条 SR 扩展

下面是 2026-06-14 继续扩大样本后的阶段性结果。为避免和 full codec ViSQOL 后台任务抢 CPU，本表当前不补 300 条 ViSQOL；Mossland、AudioSR、FlowHigh、NU-Wave2、FastWave、AERO 的 300 条 CLAP/VGGish/距离指标已补齐，UniverSR 当前只补 16/24 kHz 的 300 条 CLAP/距离指标。AERO hl=256 是 `12-48/aero-nfft=512-hl=256` checkpoint 变体，和 hl=128 主结果分开记录。

| 模型 | 任务 | 样本数 | FAD_clap | FAD_vggish | SI-SDR dB | SNR dB | LSD | LSD-LF | LSD-HF | MRSTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mossland `ckpt/mossland-codec0613` | 16 kHz -> 48 kHz | 300 | 0.389379 | 1.91660 | -13.0506 | -1.04210 | 27.0823 | 11.0471 | 32.7578 | 340.202 mean / 0.537197 median |
| Mossland `ckpt/mossland-codec0613` | 24 kHz -> 48 kHz | 300 | 0.399308 | 1.80922 | -13.0788 | -1.08349 | 29.4036 | 19.9805 | 37.5154 | 340.231 mean / 0.557360 median |
| Mossland `ckpt/mossland-codec0613` | 32 kHz -> 48 kHz | 300 | 0.396576 | 1.79800 | -13.0902 | -1.07464 | 29.2560 | 25.7763 | 36.3179 | 340.229 mean / 0.556367 median |
| FlowHigh | 16 kHz -> 48 kHz | 300 | 0.0974628 | 0.523028 | 14.7359 | 5.24391 | 43.2559 | 9.49549 | 53.5808 | 0.972326 |
| FlowHigh | 24 kHz -> 48 kHz | 300 | 0.0925670 | 0.511304 | 15.1805 | 5.39269 | 42.5037 | 22.2570 | 57.8109 | 0.966113 |
| FlowHigh | 32 kHz -> 48 kHz | 300 | 0.0910487 | 0.501722 | 15.0304 | 5.24973 | 42.9270 | 31.6377 | 63.6612 | 0.976736 |
| NU-Wave2 | 16 kHz -> 48 kHz | 300 | 0.115517 | 0.525748 | 16.2993 | 5.49372 | 41.3222 | 9.02383 | 51.1350 | 0.958072 |
| NU-Wave2 | 24 kHz -> 48 kHz | 300 | 0.0904413 | 0.503030 | 17.4865 | 5.73583 | 38.0810 | 15.9323 | 53.2899 | 0.934664 |
| NU-Wave2 | 32 kHz -> 48 kHz | 300 | 0.110399 | 0.504633 | 17.9776 | 5.81245 | 35.4909 | 21.8000 | 57.3792 | 0.921167 |
| FastWave official checkpoint | 16 kHz -> 48 kHz | 300 | 0.130382 | 0.564108 | 15.5690 | 5.29748 | 47.2015 | 10.4046 | 57.1456 | 0.983728 |
| FastWave official checkpoint | 24 kHz -> 48 kHz | 300 | 0.142833 | 0.564454 | 15.9226 | 5.39215 | 46.7266 | 20.5868 | 62.4462 | 0.974695 |
| FastWave official checkpoint | 32 kHz -> 48 kHz | 300 | 0.188266 | 0.564130 | 16.0120 | 5.40189 | 45.9800 | 28.0167 | 68.7630 | 0.969998 |
| AudioSR basic | 16 kHz -> 48 kHz | 300 | 0.0988956 | 1.01188 | 11.0552 | 5.19934 | 31.0045 | 12.0428 | 37.5498 | 0.682118 |
| AudioSR basic | 24 kHz -> 48 kHz | 300 | 0.0916977 | 0.885993 | 11.2467 | 5.26633 | 31.3427 | 22.3917 | 39.0024 | 0.681901 |
| AudioSR basic | 32 kHz -> 48 kHz | 300 | 0.0915464 | 0.881492 | 11.2810 | 5.28213 | 31.2945 | 27.9154 | 37.8979 | 0.680452 |
| UniverSR audio | 16 kHz -> 48 kHz | 300 | 0.552213 | 12.8308 | -48.6715 | -2.67489 | 37.7328 | 50.1611 | 28.1768 | 1.06475 |
| UniverSR audio | 24 kHz -> 48 kHz | 300 | 0.168322 | 7.44455 | -45.1455 | -2.69954 | 30.5195 | 37.0386 | 19.8399 | 1.02634 |
| AERO official 12-48 ckpt hl=128 | 12 kHz -> 48 kHz | 300 | 0.110126 | 0.578600 | 13.9995 | 14.4371 | 42.7750 | 6.09525 | 49.9396 | 0.286422 |
| AERO official 12-48 ckpt hl=256 | 12 kHz -> 48 kHz | 300 | 0.205793 | 3.38589 | 9.89004 | 9.96797 | 36.4117 | 11.3140 | 42.0531 | 0.355104 |

仍在后台运行的扩大任务：Demucs/MDX full-track separation `eval_none` 和 full codec ViSQOL。完成后继续补入对应任务节。

## MusicCaps-HF A2SB BWE

NVIDIA A2SB 使用官方 `NVIDIA/diffusion-audio-restoration` repo 与 HF `nvidia/audio_to_audio_schrodinger_bridge` checkpoints，当前按 44.1 kHz mono bandwidth-extension/restoration 路径运行 `predict_n_steps=50`。这不是当前 16/24/32 kHz -> 48 kHz SR 同协议，应单列为 `44.1 kHz BWE` 工程评测。

| 模型 | 任务 | 样本数 | FAD_clap | FAD_vggish | ViSQOL MOS-LQO | SI-SDR dB | SNR dB | LSD | LSD-LF | LSD-HF | MRSTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2SB official checkpoint | 16 kHz low-band -> 44.1 kHz BWE | 100 | 0.0357285 | 0.0920404 | 4.14483 | 16.0159 | 15.6913 | 20.9284 | 5.03267 | 25.8952 | 0.187575 |
| A2SB official checkpoint | 16 kHz low-band -> 44.1 kHz BWE | 300 | 0.0287543 | 0.0517306 | pending | 15.8270 | 15.6961 | 21.9842 | 5.20493 | 27.1967 | 0.191048 |

## MusicCaps-HF 20 条 Mono-to-Stereo Baselines

下面使用 `hf_musiccaps_multitask_manifest_100.jsonl` 中的 `mono_to_stereo` 子集。DiffStereo adapter 运行官方 checkpoint `model_epoch_80000.pt`、24 kHz、9.1 秒 chunk、空 `timestep_respacing` 的 1000-step sampler，输出在 `runs/baselines_musiccaps_stereo20_official/`。s3a 使用 `runs/baselines_musiccaps_stereo20_s3a/`，当前因官方 API 路径仍需依赖排查，表中是 adapter fallback DSP decorrelator 结果。二者都不是完整论文 benchmark 或主观 MOS 复现。

| 模型 | 样本数 | FAD_clap | FAD_vggish | ViSQOL MOS-LQO | SI-SDR dB | SNR dB | LSD | MRSTFT | fold-down SI-SDR | stereo width | channel corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffStereo official sampler | 20 | 0.403286 | 9.54824 | 2.85661 | 0.773159 | 0.575298 | 31.3813 | 1.5132 | 4.85469 | 0.45854 | 0.644269 |
| s3a fallback decorrelator | 20 | 0.00171721 | 0.032008 | 4.73195 | 6.84074 | 7.82609 | 5.50703 | 0.335762 | 61.4189 | 0.318647 | 0.960634 |

这些客观指标主要是诊断：`mono_to_stereo` 是多解生成任务，单个 stereo reference 的 SI-SDR/LSD 不应作为主排序依据；更应结合 fold-down 保真、宽度/相关性和主观 spatial MOS。s3a fallback 的 FAD/ViSQOL/fold-down 指标很强，主要反映它接近原始 mono/fold-down；其 `stereo_width` 与 `channel_corr` 才更能体现它作为传统 decorrelation anchor 的空间差异。

### MusicCaps-HF 100 条 Mono-to-Stereo 扩展

Ambisonizer 使用官方 `SEANet(480000, 64)` 和官方 checkpoint，输出固定 44.1 kHz、480000 samples，再按官方 notebook 的 FOA decode 方位 `[-60, 120]` / `w_gain=0.3` 解码为 stereo。DiffStereo 使用官方 checkpoint 和空 `timestep_respacing` 的 1000-step sampler；当前 seed0/seed1 100 条均已完成。s3a 当前是 fallback DSP decorrelator anchor，不是官方 API 路径。本表当前已补 CLAP/VGGish/距离/空间指标；ViSQOL 暂不启动，避免和 full codec ViSQOL 后台任务抢 CPU。

| 模型 | 样本数 | FAD_clap | FAD_vggish | SI-SDR dB | SNR dB | LSD | MRSTFT | fold-down SI-SDR | stereo width | channel corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffStereo official sampler seed0 | 100 | 0.319345 | 6.05737 | 0.805553 | 0.185681 | 31.8635 | 1.74348 | 4.91585 | 0.454957 | 0.650940 |
| DiffStereo official sampler seed1 | 100 | 0.317013 | 6.00581 | 0.963559 | 0.288608 | 31.7993 | 1.72279 | 4.95550 | 0.438714 | 0.671047 |
| Ambisonizer official checkpoint | 100 | 0.221796 | 2.27908 | 3.96101 | 2.45964 | 16.3426 | 0.647271 | 51.3382 | 0.543435 | 0.577997 |
| s3a fallback decorrelator | 100 | 0.000894388 | 0.0258258 | 6.83952 | 7.47649 | 6.15601 | 0.360536 | 60.9725 | 0.324331 | 0.954480 |

## Baseline Smoke 结果

| Baseline | 任务 | 样本数 | 结果文件 | 说明 |
| --- | --- | ---: | --- | --- |
| AudioSR | `super_resolution` | 300 | `scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000/eval_clap/summary.md` | 官方 AudioSR `basic` checkpoint 已扩到 16/24/32 kHz 三个 bucket 各 100 条；见上表。 |
| FlowHigh | `super_resolution` | 300 | `scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_flowhigh/eval_clap/summary.md` | 官方/维护 checkpoint 已扩到 16/24/32 kHz 三个 bucket 各 100 条；见上表。 |
| NU-Wave2 | `super_resolution` | 300 | `scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_nuwave2/eval_clap/summary.md` | 官方 48 kHz target checkpoint 已扩到 16/24/32 kHz 三个 bucket 各 100 条；见上表。 |
| UniverSR | `super_resolution` | 200 | `scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_universr/eval_clap/summary.md` | 官方 `woongzip1/universr-audio` checkpoint 已跑 16/24 kHz 两个官方支持 bucket 各 100 条；CLAP/VGGish/ViSQOL 均已完成。 |
| A2SB | `super_resolution` / `bandwidth_extension` | 100 | `scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_a2sb/eval_clap/summary.md` | 官方 repo/checkpoint 已接入；MusicCaps-HF 100 条 44.1 kHz BWE 工程评测已完成并单列表，协议不与当前 48 kHz SR 表混作同协议。 |
| DiffStereo | `mono_to_stereo` | 20 | `scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_stereo20_official/diffstereo_eval_clap/summary.md` | 官方 checkpoint 和 1000-step sampler 已从 `ddim25` smoke 扩到 20 条小评测；见上表。 |
| s3a | `mono_to_stereo` | 20 | `scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_stereo20_s3a/eval_clap/summary.md` | 官方 repo 已 clone，adapter fallback DSP decorrelator 已跑通 20 条并完成 CLAP/VGGish/ViSQOL。 |

## MUSDB18-HQ 10 秒 Separation

`scripts/mossland-codec/eval_benchmark/data/musdb18hq_test_10s_manifest.jsonl` 使用 MUSDB18-HQ test split 50 首歌，每首取起点 0 的 10 秒，分别评估 vocals 与 accompaniment，共 100 行。下面距离/FAD/mir_eval 结果来自 `ckpt/mossland-codec0613`，只代表 10 秒工程评测，不是 SiSEC full-track museval v4。

| 任务 | 样本数 | FAD_vggish | SI-SDR dB | SNR dB | LSD | MRSTFT | mir_eval SDR | mir_eval SAR | mir_eval ISR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| separate_vocals | 50 | 3.58069 | -74.2113 | -58.3736 | 82.3073 | 625142 | -7.60697 | -16.0326 | -1.38076 |
| separate_accompaniment | 50 | 4.69870 | -33.7850 | -16.6014 | 37.6134 | 16484.9 | -2.54682 | -6.39240 | 0.527238 |

`mir_eval_bss_images_*` 是单源 diagnostic；`FAD_vggish` 来自 `runs/mossland_musdb18hq_test_10s_vggish/summary.md`。

### MUSDB18-HQ 10 秒 museval v4

`museval_eval.py` 对同一 10 秒 manifest 按 track 同时传入 vocals/accompaniment，调用 `museval.evaluate(..., mode="v4")`。50 首里 35 首起点 0 的 10 秒片段有某个 reference source 全静音，BSS Eval v4 按规范拒绝；有效 15 首、30 个 target rows。结果文件在 `runs/mossland_musdb18hq_test_10s_museval/`，`errors.jsonl` 记录跳过 track。

| Target | 有效 track 数 | museval SDR | museval SIR | museval SAR | museval ISR |
| --- | ---: | ---: | ---: | ---: | ---: |
| vocals | 15 | -8.05049 | 1.04050 | -12.4066 | -1.53980 |
| accompaniment | 15 | -3.71643 | 5.16531 | -4.44980 | -0.141063 |

这些数值是 10 秒片段级 paired-source BSS Eval v4，不应与 SiSEC full-track median SDR 直接比较。后续已新增非静音窗口 manifest，作为当前稳定 10 秒横评口径。

### MUSDB18-HQ 10 秒非静音 frame museval v4

`build_musdb_manifest.py --select-non-silent --non-silent-frame-seconds 1` 会为每首 test track 选择一个 10 秒窗口，并要求窗口内每个 1 秒 frame 的 vocals 与 accompaniment RMS 都超过 `1e-5`，以匹配 `museval` v4 frame-wise all-zero source 约束。manifest 为 `data/musdb18hq_test_10s_nonsilent_frame_manifest.jsonl`，Mossland 输出在 `runs/mossland_musdb18hq_test_10s_nonsilent_frame_vggish/`，paired-source `museval` 输出在 `runs/mossland_musdb18hq_test_10s_nonsilent_frame_museval/`。50 首 test track 全部有效，无 `errors.jsonl`。

| 任务 | 样本数 | FAD_vggish | SI-SDR dB | SNR dB | LSD | MRSTFT | mir_eval SDR | mir_eval SAR | mir_eval ISR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| separate_vocals | 50 | 5.00457 | -33.2578 | -3.23534 | 30.2062 | 1.70890 | -3.14379 | -9.45315 | 0.297784 |
| separate_accompaniment | 50 | 2.51262 | -19.1808 | -1.64960 | 19.9120 | 0.674277 | -1.63584 | -6.23916 | 0.909554 |

| Target | 有效 track 数 | museval SDR | museval SIR | museval SAR | museval ISR |
| --- | ---: | ---: | ---: | ---: | ---: |
| vocals | 50 | -3.64439 | 3.83420 | -7.08427 | 0.189689 |
| accompaniment | 50 | -2.14869 | 9.57009 | -5.84080 | 0.649102 |

这仍是 10 秒片段级工程评测，不是 SiSEC full-track median SDR；但它比起点 0 manifest 更接近 `museval` v4 的有效输入条件，可作为后续 full-track 前的稳定回归协议。

### MUSDB18-HQ 10 秒 ByteSep ResUNet143

ByteSep 使用官方 ByteDance `music_source_separation` repo、Zenodo `ResUNet143_Subbandtime` vocals/accompaniment checkpoints 和官方 MUSDB18 config。adapter 不安装官方旧依赖栈，而是在当前环境中用轻量 `pytorch_lightning` shim 加载模型；MobileNet 小权重已先完成 1 track smoke，下面是 ResUNet143 在同一非静音 10 秒 manifest 上的 50 track 工程评测。

| 任务 | 样本数 | FAD_clap | FAD_vggish | SI-SDR dB | SNR dB | LSD | MRSTFT | mir_eval SDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| separate_vocals | 50 | 0.285021 | 0.904098 | -5.00561 | 5.23155 | 40.4360 | 1.19370 | 5.48669 |
| separate_accompaniment | 50 | 0.110299 | 0.419900 | 18.4457 | 18.5960 | 10.5010 | 0.150711 | 18.5960 |

| Target | 有效 track 数 | museval SDR | museval SIR | museval SAR | museval ISR |
| --- | ---: | ---: | ---: | ---: | ---: |
| vocals | 50 | 4.68830 | 11.6226 | 5.86783 | 12.0098 |
| accompaniment | 50 | 19.4639 | 27.2273 | 20.9069 | 27.2086 |

这仍是 10 秒片段级工程评测，不是 ByteSep/SigSep full-track 官方口径；但与 Mossland/HTDemucs/Open-Unmix 使用同一非静音 manifest，可作为当前本地横向对比。

## 下一步

1. 获取或确认官方 MusicCaps 音频来源：第三方 HF 镜像可用于工程 smoke，但正式论文复现仍需说明非官方来源；若走 YouTube，需要 cookies 给 `download_musiccaps_audio.py`。
2. 等待当前 full codec ViSQOL 后台任务完成；完成后替换 codec 表里的 `pending`。
3. 挂 Mossland separation chunked full-track 全量推理；`fulltrack_infer.py` 1-track smoke 已通过，旧 full-track estimates 因预测长度只有约 `1.509s` 已判定无效。
4. 继续接或扩大剩余开源官方 checkpoint，明确哪些是工程评测、哪些可复现论文协议；优先选择能利用空闲 GPU 且不阻塞 ViSQOL CPU 后台任务的工作。
