# 论文资料索引

本目录保存 Mossland 多任务评估指标调研用到的公开论文 PDF。阅读 `docs/evaluation-metrics.md` 时可用本索引快速定位论文。

## Codec / Reconstruction

- `encodec.pdf`：EnCodec，codec/reconstruction 评估中使用 MUSHRA、ViSQOL、SI-SNR。
- `soundstream.pdf`：SoundStream，codec 主观听测、ViSQOL、码率曲线。
- `dac_rvqgan.pdf`：DAC / improved RVQGAN，ViSQOL、mel/STFT distance、SI-SDR、MUSHRA、bitrate efficiency。
- `visqol_v3.pdf`：ViSQOL v3 指标来源和开源实现说明。
- `fad_original.pdf`：Fréchet Audio Distance 原始论文。
- `fad_music_eval.pdf`：FAD 在音乐生成评估中的 sample-size bias 和 embedding 选择问题。
- `webmushra.pdf`：webMUSHRA 框架论文，听测落地参考。

## Source Separation

- `sisec2018.pdf`：SiSEC 2018、MUSDB18、BSS Eval v4 和 `museval` 来源。
- `bss_eval_vincent.pdf`：BSS Eval 原始指标论文。
- `sdr_half_baked.pdf`：SDR 缺陷与 SI-SDR/SD-SDR 替代指标。
- `source_sep_human_vs_metrics.pdf`：BSS Eval / PEASS 与 human listening test 的相关性问题。

## Bandwidth Extension / Audio Super-resolution

- `nuwave.pdf`：NU-Wave，SNR、LSD 和 ABX/human distinguishability。
- `nuwave2.pdf`：NU-Wave 2，LSD-HF / LSD-LF 分解。
- `audiosr.pdf`：AudioSR，LSD 与主观质量不一致的案例、MOS/preference。
- `aero.pdf`：AERO，LSD、ViSQOL、MUSHRA。

## Stereo / Spatial Generation

- `diffstereo.pdf`：DiffStereo，mono-to-stereo generation，LSD/SI-SNR/FAD/wideness/MOS。
- `parametric_stereo_generation.pdf`：Parametric stereo generation，明确 mono-to-stereo 是多解主观任务。
- `binauralgrad.pdf`：BinauralGrad，Wave/Amplitude/Phase L2、PESQ、MRSTFT、MOS/Similarity/Spatial MOS。
- `beyond_mono_binaural.pdf`：Beyond Mono to Binaural，ILD/ITD 背景和 STFT/ENV/Mag/Phs/SNR。
- `sep_stereo.pdf`：Sep-Stereo，STFTD/ENVD 和 stereo user study。
- `diff_sage.pdf`：Diff-SAGe，空间音频生成的 condition/distribution/subjective 指标。
