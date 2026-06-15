# Data Processing

这里放面向数据处理任务的轻量入口。

## OSS Separation Queue

`oss_prepare_separation.py` 把一条 OSS 音频处理成 RoFormer separation 产物：

1. 从 OSS 下载单个音频到 `local_queue` 分配的 worker 临时目录。
2. 用 `ffprobe` 预读本地临时音频/视频容器时长，失败时回退 `torchaudio.info()`；默认大于等于 600 秒时不进入 RoFormer，上传 `metadata.json(status=skiplong, reason=max_duration_exceeded)`。
3. 短音频调用 `scripts.data_processing.separation_core` 生成 `mixture.mp3`、`vocals.mp3`、`accompaniment.mp3`、`metadata.json`；每个 worker 进程缓存一个 RoFormer separator，不按单曲重复加载模型。
4. 上传四个文件到对应 OSS 输出目录。
5. 删除该任务的本地临时音频和 stem 文件，只保留给 `local_queue` commit 的 receipt。

支持的输入后缀包括 `.aac/.aif/.aiff/.flac/.m4a/.mp3/.mp4/.ogg/.opus/.webm/.wav`。`.webm/.mp4` 走 `ffmpeg` 抽音频流。模型输入按模型 config 的 `sample_rate` 统一重采样；输出三个 stem 固定保存为 44100 Hz。只有源文件是 44100 Hz、声道数匹配的 `.mp3` 时，`mixture.mp3` 会直接复制源文件，否则重采样并重新编码。

配置示例：

```json
{
  "inputs_file": "oss_inputs.txt",
  "source_prefix": "qz_oss2:public/raw_audio",
  "output_prefix": "qz_oss2:public/prepared_separation",
  "config_hash": "separation_v1",
  "model_dir": "checkpoints/mel-band-roformer-vocal-model",
  "device": "cuda:0",
  "chunk_batch_size": 1,
  "max_duration_seconds": 600,
  "oss_timeout_seconds": 3600
}
```

`oss_inputs.txt` 一行一个 OSS 路径。也可以用配置字段 `inputs` 直接传列表。
`max_duration_seconds` 省略时默认为 600；设为 `0` 或负数可关闭 OSS handler 的时长预读跳过。
远端已有完整四个产物时 receipt 状态为 `skipped`；远端已有 `metadata.json(status=skiplong)` 或本轮判定超长时，receipt 状态为 `skiplong`，不是普通 skip。

2026-06-12 PPU 实测 100 条 `YouTube_Music` 混合样本：60 `.mp3`、20 `.webm`、20 `.mp4`，8 worker 用 181 秒完成；receipt 状态为 99 `done`、1 `skiplong`、0 failed。抽查 `.webm` 和 `.mp4` 输出的 `mixture/vocals/accompaniment` 都是 44100 Hz、2 声道。

`oss_prepare_separation_pipeline.py` 是同一 queue 格式的高利用率 worker。普通 `local_queue worker` 在单卡内按“下载/预读 -> GPU 推理 -> 保存 -> 上传”串行执行；pipeline worker 会在同一 GPU 进程里提前下载/预读后续任务，同时把当前任务的 MP3 保存和 OSS 上传放到后台线程，GPU 主线程继续处理下一条。默认参数：

- `--prefetch 4`：每张 GPU 最多持有 4 个已 lease/下载中的任务。
- `--download-workers 2`：每张 GPU 2 个下载/预读线程。
- `--upload-workers 2`：每张 GPU 2 个保存/上传后台线程。
- `--max-pending-uploads 2`：最多缓存 2 个待保存/上传结果，防止 CPU 内存堆积。

`bash/prepare_youtubemusic_seperation/03_start_workers.sh` 默认使用 pipeline worker。若要回退旧串行 worker，可设置 `USE_PIPELINE_WORKER=0`。

初始化和运行：

```bash
python -m scripts.tools.local_queue.cli init \
  --root /shared/local_queue \
  --job netease_separation_v1 \
  --handler scripts.data_processing.oss_prepare_separation:OssPrepareSeparationHandler \
  --config config.json

python -m scripts.tools.local_queue.cli worker \
  --root /shared/local_queue \
  --job netease_separation_v1 \
  --handler scripts.data_processing.oss_prepare_separation:OssPrepareSeparationHandler \
  --device cuda:0
```

pipeline worker 单卡示例：

```bash
python -m scripts.data_processing.oss_prepare_separation_pipeline \
  --root /shared/local_queue \
  --job netease_separation_v1 \
  --device cuda:0 \
  --prefetch 4 \
  --download-workers 2 \
  --upload-workers 2 \
  --max-pending-uploads 2
```
