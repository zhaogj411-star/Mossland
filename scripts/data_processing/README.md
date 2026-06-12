# Data Processing

这里放面向数据处理任务的轻量入口。

## OSS Separation Queue

`oss_prepare_separation.py` 把一条 OSS 音频处理成 RoFormer separation 产物：

1. 从 OSS 下载单个音频到 `local_queue` 分配的 worker 临时目录。
2. 调用 `scripts.data.prepare_separation.process_file()` 生成 `mixture.mp3`、`vocals.mp3`、`accompaniment.mp3`、`metadata.json`。
3. 上传四个文件到对应 OSS 输出目录。
4. 删除该任务的本地临时音频和 stem 文件，只保留给 `local_queue` commit 的 receipt。

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
  "oss_timeout_seconds": 3600
}
```

`oss_inputs.txt` 一行一个 OSS 路径。也可以用配置字段 `inputs` 直接传列表。

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
