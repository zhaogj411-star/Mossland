# Mossland Eval Benchmark

本目录保存 `scripts/mossland-codec/tasks.py` 五个任务的评估 pipeline。

## 入口

从 prepared separation 目录生成固定 manifest：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.build_manifest \
  --prepared-root /path/to/prepared/root \
  --output scripts/mossland-codec/eval_benchmark/data/mossland_eval.jsonl \
  --max-items 300 \
  --duration-seconds 10 \
  --sr-rates 8000,12000,16000,24000,32000,40000 \
  --stereo-seeds 0,1,2,3
```

从 CoDiCodec 使用的 MusicCaps metadata 生成 codec/reconstruction manifest：

```sh
# metadata 已下载到 data/musiccaps；若需重建：
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://huggingface.co \
  python - <<'PY'
from pathlib import Path
from datasets import load_dataset
out_dir = Path("scripts/mossland-codec/eval_benchmark/data/musiccaps")
out_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset("google/MusicCaps", split="train", cache_dir=str(out_dir / "hf_cache"))
ds.to_json(str(out_dir / "musiccaps_train_metadata.jsonl"), force_ascii=False)
PY
```

下载并裁切 MusicCaps YouTube 音频。当前网络直连 YouTube 不通，代理可访问但需要 YouTube cookies，否则会触发 bot 验证：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.download_musiccaps_audio \
  --metadata scripts/mossland-codec/eval_benchmark/data/musiccaps/musiccaps_train_metadata.jsonl \
  --audio-dir scripts/mossland-codec/eval_benchmark/data/musiccaps/audio \
  --success-output scripts/mossland-codec/eval_benchmark/data/musiccaps/download_success.jsonl \
  --failure-output scripts/mossland-codec/eval_benchmark/data/musiccaps/download_failure.jsonl \
  --use-env-proxy \
  --cookies /path/to/youtube-cookies.txt \
  --max-items 300
```

生成 CoDiCodec-style reconstruction manifest：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.build_musiccaps_manifest \
  --metadata scripts/mossland-codec/eval_benchmark/data/musiccaps/musiccaps_train_metadata.jsonl \
  --audio-root scripts/mossland-codec/eval_benchmark/data/musiccaps/audio \
  --output scripts/mossland-codec/eval_benchmark/data/musiccaps/musiccaps_reconstruct_manifest.jsonl \
  --missing-output scripts/mossland-codec/eval_benchmark/data/musiccaps/musiccaps_missing_audio.jsonl
```

从已有单音频 manifest 派生 reconstruction、super-resolution bucket 和 mono-to-stereo 多 seed 样本，适合把 MusicCaps-HF/MUSDB mixture 等音频清单扩展成非 separation 任务评测：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.build_audio_task_manifest \
  --input scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_reconstruct_manifest.jsonl \
  --output scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_multitask_manifest_10.jsonl \
  --max-items 10 \
  --tasks reconstruct,super_resolution,mono_to_stereo \
  --sr-rates 16000,24000,32000 \
  --stereo-seeds 0,1
```

可选：从第三方 Hugging Face MusicCaps 音频镜像下载 wav。这个源不是 Google 官方再分发，正式报告里必须标注为 `mahendra0203/musiccaps_processed_full` 镜像并确认许可：

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://huggingface.co \
  HF_HOME=$PWD/scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_cache \
  HF_DATASETS_CACHE=$PWD/scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_datasets_cache \
  PYTHONPATH=$PWD \
  python -m scripts.mossland-codec.eval_benchmark.download_musiccaps_hf_audio \
    --audio-dir scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_audio \
    --success-output scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_download_success.jsonl \
    --failure-output scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_download_failure.jsonl \
    --max-items 0 \
    --progress-every 100
```

下载公开 separation 评测集。当前 `public_datasets.json` 注册了 MUSDB18-HQ 和 MUSDB18；Zenodo 在本机绕过代理可访问：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.download_public_dataset \
  --dataset musdb18hq \
  --head \
  --no-proxy

PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.download_public_dataset \
  --dataset musdb18hq \
  --no-proxy \
  --progress-seconds 30
```

下载器会写 `.part` 临时文件、复用断点，并在完成时校验 registry 里的文件大小。MUSDB18-HQ 解压后可生成 separation manifest：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.build_musdb_manifest \
  --musdb-root scripts/mossland-codec/eval_benchmark/data/musdb18hq \
  --output scripts/mossland-codec/eval_benchmark/data/musdb18hq_test_manifest.jsonl \
  --split test
```

如果要跑 10 秒片段级 `museval` v4，建议选择每个 1 秒 BSS Eval frame 内 vocals/accompaniment 都非静音的窗口，避免被 `museval` 的 all-zero source 约束拒绝：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.build_musdb_manifest \
  --musdb-root scripts/mossland-codec/eval_benchmark/data/musdb18hq \
  --output scripts/mossland-codec/eval_benchmark/data/musdb18hq_test_10s_nonsilent_frame_manifest.jsonl \
  --split test \
  --duration-seconds 10 \
  --select-non-silent \
  --window-hop-seconds 5 \
  --min-source-rms 1e-5 \
  --non-silent-frame-seconds 1
```

运行 checkpoint 推理和指标：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.run \
  --manifest scripts/mossland-codec/eval_benchmark/data/example_manifest.jsonl \
  --checkpoint-dir ckpt/mossland-codec0613 \
  --device cuda \
  --metrics-device cuda \
  --fad-backend mel_proxy \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/mossland-codec0613
```

`run.py` 的模型和 STFT/CLAP 指标可分别指定设备。默认会复用确定性输出路径下已有的非空 prediction wav，方便长评估中断恢复；需要强制重算时传 `--overwrite-predictions`。如果 manifest 已经包含 `prediction_path`，可以省略 `--checkpoint-dir`，只计算指标。长音频或大 manifest 可用 `--num-shards/--shard-id` 拆分到多个独立 `--output-dir`，并用 `--progress-every` 在 stderr 打印进度；每个 shard 的 FAD 只对 shard 内样本成立，最终论文表需要用合并后的全量 `results.jsonl` 重新聚合 FAD。

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.run \
  --manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bytesep_resunet/bytesep_resunet_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bytesep_resunet/eval_clap_shard0of4 \
  --fad-backend clap \
  --metrics-device cuda \
  --num-shards 4 \
  --shard-id 0 \
  --progress-every 1
```

合并分片结果并重新聚合全量 summary：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.aggregate_results \
  --results \
    scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bytesep_resunet/eval_clap_shard0of4/results.jsonl \
    scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bytesep_resunet/eval_clap_shard1of4/results.jsonl \
    scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bytesep_resunet/eval_clap_shard2of4/results.jsonl \
    scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bytesep_resunet/eval_clap_shard3of4/results.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bytesep_resunet/eval_clap_merged
```

高吞吐后评测优先使用封装入口，避免手工启动单进程全量任务。GPU 指标用 `run_sharded_metric_eval.py`，它会按 GPU 列表分片运行 `run.py`、限制每进程 CPU library 线程，并在最后调用 `aggregate_results.py` 重新计算全量 FAD：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.run_sharded_metric_eval \
  --manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_full_dac_nq2/dac_nq2_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_full_dac_nq2/eval_clap \
  --shard-output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_full_dac_nq2/eval_clap_shards \
  --fad-backend clap \
  --gpus 0,1,2,3,4,5,6,7 \
  --num-shards 8 \
  --progress-every 100
```

ViSQOL 是 CPU-only，不能放到 GPU 上。大 manifest 使用 `run_sharded_visqol_eval.py` 多进程分片，并根据机器负载调 `--num-shards`；本机 5355 条 MusicCaps-HF 三档 EnCodec 曾用 64 shards/码率并行完成：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.run_sharded_visqol_eval \
  --manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_full_encodec_bw6/encodec_bw6_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/visqol_full_codec/encodec_bw6 \
  --shard-output-dir scripts/mossland-codec/eval_benchmark/runs/visqol_full_codec/encodec_bw6_shards64 \
  --num-shards 64
```

从当前本地 `summary.json` / ViSQOL CSV 产物刷新复现表：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.generate_tables
```

默认输出到 `scripts/mossland-codec/eval_benchmark/tables/`，包含 Markdown 和 CSV 两种格式。该脚本只汇总已有实验产物，不重新推理；表格只保留模型、数据/任务和指标列。

SemantiCodec full 5355 推理较慢，adapter 只在推理结束时写 final manifest。可用 watcher 等三档 `semanticodec_manifest.jsonl` 完整后自动跑 CLAP/VGGish 指标并刷新表格：

```sh
setsid bash scripts/mossland-codec/eval_benchmark/watch_semanticodec_full_metrics.sh \
  > scripts/mossland-codec/eval_benchmark/logs/semanticodec_full_metrics_watcher.log 2>&1 < /dev/null &
```

默认使用 PPU/GPU `3,4,5` 分别评 token-rate `25,50,100`；可用 `SEMANTICODEC_DEVICE_TR25/TR50/TR100` 覆盖。

运行官方 baseline adapter。已接入 Facebook Research EnCodec、Descript DAC、SNAC、WavTokenizer、CoDiCodec、AudioSR、NU-Wave2、FlowHigh、UniverSR、AERO、DiffStereo、s3a、HTDemucs/Demucs family、Open-Unmix/UMXHQ/UMXL、ByteSep 和 ZFTurbo/MVSep Music-Source-Separation-Training 的本地 adapter；adapter 输出 prediction manifest 后可继续交给 `run.py` 算 Mossland 统一指标。EnCodec 示例：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.encodec_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musiccaps/musiccaps_reconstruct_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines/encodec_manifest.jsonl \
  --device cuda \
  --bandwidth 24
```

DAC reconstruction 示例：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.dac_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_reconstruct_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines/dac_manifest.jsonl \
  --device cuda \
  --weights-path tmp/eval_baseline_refs/dac/checkpoints/weights.pth
```

AudioSR super-resolution 示例。默认使用官方 `basic` checkpoint、50 DDIM steps、guidance 3.5；`docs/eval-benchmark-report.md` 里已有 MusicCaps-HF SR@16000 前 10 条小评测结果：

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://huggingface.co \
  PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
  python -m scripts.mossland-codec.eval_benchmark.baselines.sr_baseline \
    --manifest scripts/mossland-codec/eval_benchmark/data/tmp_34078087_manifest.jsonl \
    --output-dir tmp/eval_baseline_refs/sr/smoke_eval \
    --output-manifest tmp/eval_baseline_refs/sr/smoke_eval/audiosr_manifest.jsonl \
    --device cuda \
    --max-items 1
```

DiffStereo mono-to-stereo 示例。默认空 `--timestep-respacing` 是官方 1000-step sampler；`ddim25` 只适合 smoke。注意使用小写 `tmp/eval_baseline_refs/diffstereo/...` 下真实 checkpoint；大写 `tmp/eval_baseline_refs/DiffStereo/...` 里的同名文件可能只是 Git LFS pointer。`docs/eval-benchmark-report.md` 里已有 MusicCaps-HF seed0 前 20 条 official sampler 小评测结果：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.diffstereo_baseline \
  --manifest tmp/eval_baseline_refs/diffstereo/smoke/manifest.jsonl \
  --output-dir tmp/eval_baseline_refs/diffstereo/smoke/preds \
  --output-manifest tmp/eval_baseline_refs/diffstereo/smoke/pred_manifest.jsonl \
  --repo-dir tmp/eval_baseline_refs/diffstereo/DiffStereo \
  --checkpoint tmp/eval_baseline_refs/diffstereo/DiffStereo/checkpoints/model_epoch_80000.pt \
  --device cuda \
  --timestep-respacing ddim25
```

CoDiCodec reconstruction 示例。官方仓库当前不带评估脚本，本 adapter 只负责调用官方 checkpoint 推理；指标仍由本目录 `run.py` 计算：

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://huggingface.co \
  PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
  python -m scripts.mossland-codec.eval_benchmark.baselines.codicodec_baseline \
    --manifest scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_reconstruct_manifest_100.jsonl \
    --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_reconstruct100_codicodec \
    --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_reconstruct100_codicodec/codicodec_manifest.jsonl \
    --repo-dir tmp/codicodec \
    --device cuda:0
```

NU-Wave2 和 FlowHigh super-resolution 示例。二者输出 48 kHz mono prediction；FlowHigh 官方路径当前需要 CUDA：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.nuwave2_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_sr16000_manifest_100.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_nuwave2 \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_nuwave2/nuwave2_manifest.jsonl \
  --repo-dir tmp/eval_baseline_refs/nuwave2 \
  --checkpoint tmp/eval_baseline_refs/nuwave2/checkpoints/nuwave2_48k_official.ckpt \
  --device cuda

PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland:tmp/eval_baseline_refs/flowhigh/flowhigh/src \
python -m scripts.mossland-codec.eval_benchmark.baselines.flowhigh_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_sr16000_manifest_100.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_flowhigh \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_flowhigh/flowhigh_manifest.jsonl \
  --repo-dir tmp/eval_baseline_refs/flowhigh/flowhigh \
  --checkpoint-dir checkpoints/flowhigh \
  --device cuda
```

UniverSR super-resolution 示例。官方 `woongzip1/universr-audio` 支持 `8/12/16/24 kHz -> 48 kHz`，不要把 32 kHz bucket 写成官方协议结果。adapter 使用 48 kHz reference 先 downsample/up 到有效低采样率，再调用官方 `UniverSR.enhance(..., input_sr=...)`：

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  HF_ENDPOINT=https://huggingface.co \
  huggingface-cli download woongzip1/universr-audio --local-dir checkpoints/universr-audio

PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.universr_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_sr16000_manifest_100.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_universr \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_16000_universr/universr_manifest.jsonl \
  --repo-dir tmp/eval_baseline_refs/universr/UniverSR \
  --checkpoint-dir checkpoints/universr-audio \
  --device cuda \
  --ode-method midpoint \
  --ode-steps 4 \
  --guidance-scale 1.5
```

AERO super-resolution 示例。官方 checkpoint 里有 `12-48` 权重；官方 repo 当前只带 `4-16` hydra config，adapter 复用同结构并覆盖 `lr_sr=12000/hr_sr=48000`。当前 PyTorch 2.6+ 加载官方老 checkpoint 需要可信 `weights_only=False`，adapter 内部已处理：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.aero_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musiccaps/hf_musiccaps_sr16000_manifest_100.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_12000_aero \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musiccaps_sr100_12000_aero/aero_manifest.jsonl \
  --repo-dir tmp/eval_baseline_refs/aero \
  --checkpoint tmp/eval_baseline_refs/aero/checkpoints/12-48/aero-nfft=512-hl=128/checkpoint.th \
  --device cuda \
  --input-sample-rate 12000 \
  --max-items 100
```

HTDemucs / Open-Unmix separation 示例。当前报告使用 MUSDB18-HQ test 的非静音 10 秒窗口，属于工程评测；SiSEC full-track 复现需要另跑完整 track estimates 和官方 aggregation：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.separation_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musdb18hq_test_10s_nonsilent_frame_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_demucs \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_demucs/demucs_manifest.jsonl \
  --model htdemucs \
  --device cuda

PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.openunmix_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musdb18hq_test_10s_nonsilent_frame_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_openunmix \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_openunmix/openunmix_manifest.jsonl \
  --model umxhq \
  --device cuda:0
```

ByteSep separation 示例。adapter 使用官方 ByteDance `music_source_separation` repo、Zenodo checkpoint 和 MUSDB18 配置；默认 `ResUNet143_Subbandtime` 对齐官方 CLI 默认模型，`MobileNet_Subbandtime` 只适合 smoke。为避免污染主环境，adapter 直接把官方 repo 加到 `sys.path`，并用轻量 `pytorch_lightning` shim 加载模型：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.baselines.bytesep_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musdb18hq_test_10s_nonsilent_frame_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_bytesep_resunet \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_bytesep_resunet/bytesep_resunet_manifest.jsonl \
  --repo-dir tmp/eval_baseline_refs/bytesep/music_source_separation \
  --checkpoint-dir tmp/eval_baseline_refs/bytesep/checkpoints \
  --bytesep-home tmp/eval_baseline_refs/bytesep/home \
  --model-type ResUNet143_Subbandtime \
  --device cuda \
  --download-missing \
  --no-proxy
```

ZFTurbo/MVSep Music-Source-Separation-Training separation 示例。`msst_baseline.py` 调官方 `inference.py`，当前 CLI 支持 `bs_roformer`、`mel_band_roformer` 与 `scnet`；默认从本地 `tmp/eval_baseline_refs/music-source-separation-training` 找官方 repo、config 和 checkpoint，`--download-missing` 注册了 BS-RoFormer v1.0.12 与 SCNet XL IHF v1.0.15 release assets。MVSep/UVR 模型 zoo 的 checkpoint/config 来源和许可较复杂，真实论文/榜单复现前必须记录具体权重、config、commit 和是否使用额外训练数据。

```sh
git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training \
  tmp/eval_baseline_refs/music-source-separation-training

PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
CUDA_VISIBLE_DEVICES=0 \
python -m scripts.mossland-codec.eval_benchmark.baselines.msst_baseline \
  --manifest scripts/mossland-codec/eval_benchmark/data/musdb18hq_test_fulltrack_manifest.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bs_roformer \
  --output-manifest scripts/mossland-codec/eval_benchmark/runs/baselines_musdb18hq_fulltrack_bs_roformer/bs_roformer_manifest.jsonl \
  --repo-dir tmp/eval_baseline_refs/music-source-separation-training \
  --model-type bs_roformer \
  --config-path tmp/eval_baseline_refs/music-source-separation-training/configs/config_musdb18_bs_roformer.yaml \
  --checkpoint tmp/eval_baseline_refs/music-source-separation-training/checkpoints/model_bs_roformer_ep_17_sdr_9.6568.ckpt \
  --device cuda \
  --progress-every 1
```

## Manifest 字段

每行一条 JSON：

```json
{
  "item_id": "song001_10s",
  "task_id": "super_resolution",
  "source_path": "/path/to/mixture.mp3",
  "reference_path": "/path/to/mixture.mp3",
  "start_seconds": 30.0,
  "duration_seconds": 10.0,
  "sample_rate": 44100,
  "low_sample_rate": 16000,
  "seed": 0
}
```

Separation 任务可用 `mixture_path`、`vocals_path`、`accompaniment_path`。`reference_path` 缺省时，`separate_vocals` 使用 `vocals_path`，`separate_accompaniment` 使用 `accompaniment_path`，其他任务使用 `source_path`。

## 当前指标

- 通用逐样本：`SNR`、`SI-SDR`、`LSD`、`MRSTFT`。
- `super_resolution`：额外按 `low_sample_rate / 2` 切分 `LSD-LF`、`LSD-HF`，并统计高频能量占比。
- `mono_to_stereo`：额外统计 fold-down `SI-SDR`、stereo width、channel correlation。
- `separate_vocals` / `separate_accompaniment`：额外统计 `mir_eval_bss_images_sdr_db/isr_db/sar_db`。这是轻量单源 BSS diagnostic，不等同于 paired-source `museval` v4。
- `museval_eval.py`：paired-source BSS Eval v4 wrapper。它读取已有 separation `results.jsonl`，把同一 track 的 vocals/accompaniment 预测成对传给 `museval.evaluate(..., mode="v4")`；内部只在当前进程安装 `imp.reload` shim，避免改 site-packages。MUSDB18-HQ 起点 0 的 10 秒 manifest 中大量片段有某个 source 全静音，会被 BSS Eval v4 正确拒绝；片段级工程评测应优先使用 `build_musdb_manifest.py --select-non-silent --non-silent-frame-seconds 1` 生成窗口，并在报告中写有效 track 数。
- FAD：`--fad-backend mel_proxy|clap|vggish|none`。`fad_mel_proxy` 只用于流程测试和回归测试，不是论文官方 FAD；正式复现必须把后端结果单独标为 `fad_vggish`、`fad_clap` 或其他明确 embedding 名，不要混用。
- `fad_clap`：已接入 `--fad-backend clap`，通过 Hugging Face `laion/clap-htsat-unfused` 的 audio encoder 抽 CLAP embedding，模型缓存目录默认 `checkpoints/clap`。当前环境需要清掉 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 并设置 `HF_ENDPOINT=https://huggingface.co` 才能下载；CLAP 权重和 512 维 embedding smoke 已通过。
- `fad_vggish`：已接入 `--fad-backend vggish`，使用 `tmp/eval_metric_refs/torchvggish` 的 PyTorch VGGish port 和 raw 128-D embedding，对齐 Google FAD 使用 pretrained VGGish feature 的口径。权重缓存到 `checkpoints/vggish/torch_hub/checkpoints/vggish-10086976.pth`；该实现避开当前主环境缺失 TensorFlow/Beam 的问题，但报告中必须标注后端为 `torchvggish_raw`。
- ViSQOL：`visqol_eval.py` 是外部 binary wrapper，不塞进主 `run.py`。它读取带 `prediction_path` 的 manifest/results JSONL，把 reference/prediction 对齐并转成 48 kHz mono 16-bit WAV，调用 Google ViSQOL audio mode 输出 `visqol_moslqo`。stereo/upmix 任务会被 downmix，只能评价单声道音频质量，不能评价空间感。

ViSQOL wrapper 示例。ViSQOL 已在本机用 Bazel 5.1.0 构建到 `tmp/eval_metric_refs/visqol/bazel-bin/visqol`；直接执行 binary 时必须显式传源码里的 audio SVR model，wrapper 默认已设置：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.visqol_eval \
  --manifest scripts/mossland-codec/eval_benchmark/runs/mossland_musiccaps_multitask100_vggish/results.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/mossland_musiccaps_multitask100_visqol_reconstruct \
  --tasks reconstruct \
  --visqol-bin tmp/eval_metric_refs/visqol/bazel-bin/visqol
```

museval v4 wrapper 示例：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.museval_eval \
  --manifest scripts/mossland-codec/eval_benchmark/runs/mossland_musdb18hq_test_10s_vggish/results.jsonl \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/mossland_musdb18hq_test_10s_museval \
  --keep-going
```

`--metrics-device` 控制 STFT/LSD/MRSTFT/FAD embedding 的计算设备；不传时有 CUDA 默认用 `cuda`。checkpoint 推理由 `--device` 控制，模型只加载一次并常驻设备。

`scripts/mossland-codec/eval_benchmark/data/` 和 `runs/` 被 `.gitignore` 忽略，用于本地大数据、checkpoint 输出和临时 receipts。

## CoDiCodec FAD 口径

`docs/papers/codicodec.pdf` 的 audio quality 表使用 MusicCaps 作为 evaluation dataset，并同时报告：

- `SI-SDR`
- `ViSQOL`
- `FAD`：基于 pretrained VGGish feature 的 Frechet Audio Distance。
- `FAD_clap`：基于 CLAP feature 的 FAD 变体，论文说明它更符合音频质量的人类感知相关性。

本地 `tmp/codicodec` 官方仓库目前只包含 codec 推理代码，没有评估脚本或 FAD/CLAP 实现。后续复现 CoDiCodec 表格时，需要固定 MusicCaps manifest、VGGish/CLAP checkpoint、特征抽取采样率与聚合方式，并在报告里同时给出 continuous/discrete、AR/parallel decoding 设置。当前已完成 MusicCaps metadata、CLAP backend 和 PyTorch VGGish backend；MusicCaps 官方音频仍需 cookies 或内部镜像，当前 5355 条音频来自第三方 HF 镜像。

## Smoke

已用本地 `tmp/34078087` prepared stems 生成 1 秒五任务 manifest：

```sh
python -m scripts.mossland-codec.eval_benchmark.build_manifest \
  --prepared-root tmp/34078087 \
  --output scripts/mossland-codec/eval_benchmark/data/tmp_34078087_manifest.jsonl \
  --max-items 1 \
  --duration-seconds 1 \
  --sr-rates 16000 \
  --stereo-seeds 0
```

并验证目标 checkpoint 可在 GPU 上跑通五任务：

```sh
python -m scripts.mossland-codec.eval_benchmark.run \
  --manifest scripts/mossland-codec/eval_benchmark/data/tmp_34078087_manifest.jsonl \
  --checkpoint-dir ckpt/mossland-codec0613 \
  --device cuda \
  --metrics-device cuda \
  --fad-backend none \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/smoke_tmp_34078087
```

`run.py` 会同时写：

- `results.jsonl`
- `summary.json`
- `summary.csv`
- `summary.md`

## 飞书 reconstruction 复现表

飞书文档 `Multi-task audio codec` 的 reconstruction benchmark 表可由本地 summary 生成，默认输出：

- `tables/feishu_reconstruction_repro.csv`
- `tables/feishu_reconstruction_repro.md`
- `tables/feishu_reconstruction_repro.xml`

命令：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.generate_feishu_reconstruction_table
```

该表只填本地已经完成的 MusicCaps-HF reconstruction 指标。当前已覆盖 Music2Latent、Stable Audio 3 SAME-S/SAME-L、CoDiCodec official checkpoint、CoDiCodec paper-repro 10k/18k raw/EMA、Mossland codec 370k EMA、DAC 多 bitrate、EnCodec 3/6/12/24 kbps 和 Opus 8/14/24 kbps 的 `SI-SDR`、ViSQOL、`FAD_clap`、VGGish `FAD`，并按用户要求把飞书原表 `Mel distance` / `STFT distance` 列分别用本地 pipeline 的 `lsd/mean` / `mrstft/mean` 填入；这两列应解释为本地工程 proxy 指标。未实际复现的模型或指标写 `未复现` 或 `—`。当前数据源是第三方 `mahendra0203/musiccaps_processed_full` 镜像的 5355 条 MusicCaps 音频，不是 Google/YouTube 官方音频完整复现。

授权后可用生成的 XML 替换飞书表格 block。注意 `block_replace` 后 block id 会变化；2026-06-15 最新 reconstruction 表 block id 是 `doxcn9TUoVJ2bs8AQW6yHNa1zfg`。本次只定向更新 Mossland 370k EMA reconstruction 行为 `EncoderDecoder parallel decode` 指标，写入后读回确认旧 `full-clip chunked xfade` 文本消失，新指标 `-9.930/3.013/28.207/0.522/0.098/0.490` 已落表；随后从飞书新表反向同步本地 `feishu_reconstruction_repro.{csv,md,xml}`，避免本机缺少部分 summary 时全表生成导致已复现 baseline 回退为 `未复现`。

```sh
HOME=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/tools/lark-cli-home \
/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/tools/bin/lark-cli docs +update \
  --api-version v2 \
  --as user \
  --doc DC6cdohbXoGpyjx01wac1gDynHF \
  --command block_replace \
  --block-id doxcn9TUoVJ2bs8AQW6yHNa1zfg \
  --content @scripts/mossland-codec/eval_benchmark/tables/feishu_reconstruction_repro.xml \
  --format json
```

## 飞书非 reconstruction 复现表

非重建三任务的飞书表由本地 summary 生成，默认输出：

- `tables/feishu_source_separation_repro.{csv,md,xml}`
- `tables/feishu_super_resolution_repro.{csv,md,xml}`
- `tables/feishu_mono_to_stereo_repro.{csv,md,xml}`
- `tables/feishu_non_reconstruction_repro.xml`

命令：

```sh
PYTHONPATH=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/Mossland \
python -m scripts.mossland-codec.eval_benchmark.generate_feishu_non_reconstruction_tables
```

组合 XML 可追加到飞书文档末尾：

```sh
HOME=/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/tools/lark-cli-home \
/inspire/sj-ssd3/project/embodied-multimodality/public/zhaoguojie/tools/bin/lark-cli docs +update \
  --api-version v2 \
  --as user \
  --doc DC6cdohbXoGpyjx01wac1gDynHF \
  --command append \
  --content @scripts/mossland-codec/eval_benchmark/tables/feishu_non_reconstruction_repro.xml \
  --format json
```

2026-06-14 已追加成功：写入返回 revision `725`，读回 section revision `726`；新章节 block `doxcnSqNNtjWbikUpjnHpefVsLh` 包含三张表，`<tr>` 总数 `50`，与三份本地 CSV 的表头加数据行总数一致。

## 复现状态

本目录先提供 Mossland checkpoint 的统一评估框架。EnCodec、DAC、SNAC、WavTokenizer、CoDiCodec 已完成 MusicCaps-HF reconstruction full baseline；EnCodec/DAC/CoDiCodec 的 full 5355 条 ViSQOL 已补齐并写入 summary。AudioSR、FlowHigh、NU-Wave2、FastWave 已扩到三个 SR bucket 各 300 条；UniverSR 已完成 16/24 kHz 两个官方支持 bucket 各 300 条；AERO 已完成官方 12 kHz -> 48 kHz checkpoint 的 300 条推理和 CLAP/VGGish/距离指标；A2SB 已完成 300 条 BWE 推理和指标；DiffStereo、s3a、Ambisonizer 已完成 MusicCaps-HF mono-to-stereo 100 条工程评测；HTDemucs、Open-Unmix、ByteSep、Demucs MDX/MDX Extra、HTDemucs-MMI、BS-RoFormer 和 SCNet XL IHF 已完成 MUSDB18-HQ full-track 工程评测。CoDiCodec 风格 `fad_clap`、VGGish `fad_vggish`、ViSQOL audio-mode MOS-LQO 和 MUSDB paired-source museval v4 工程评测已可运行。当前未继续运行 SemantiCodec full 或非 reconstruction 后台任务；复现表只保留当前已落盘的模型与指标值。`docs/evaluation-metrics.md` 中列出的 NU-Wave 本地已有官方仓库副本但仍缺完整 checkpoint/环境复现。后续每接入一个论文 baseline，应新增独立后端模块、记录 commit/ckpt/dataset，并在报告中说明是否复现论文表格数值。
