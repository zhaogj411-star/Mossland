# Mossland Codec

本目录是 Mossland 多任务 codec 的自包含实现目录。

由于普通 `from ... import ...` 语法不能书写带 `-` 的包名，跨包引用使用
`importlib.import_module("scripts.mossland-codec...")`；包内文件之间使用相对导入。

## 实现边界

- `scripts/codicodec/` 保留为独立参考实现，`scripts/mossland-codec/` 不再 import 它。
- 本目录不再保留 `hparams.py` 或 `hparams_inference.py`。模型结构、STFT/归一化、sigma 调度和推理默认项都从构造参数传入；训练时以 Hydra `model:` 配置为主入口，推理时使用 `EncoderDecoder(model_kwargs=...)`。
- `models.py` 提供 `MosslandCodecTransformer`，decoder conditioning 只使用 sigma embedding 和 5 个任务的 `task_embedding`，并负责 `prepare_audio_batch()` 与 `generate_waveform()`。
- `wrapper.py` 提供 `MosslandCodecTrainingWrapper`，自包含 optimizer、EMA、sigma sampling、consistency loss 和训练 step；它只消费 `MosslandTaskDataset` 生成的任务 payload，不抽样任务、不构造 `src/target`。
- `tasks.py` 提供 `MosslandTaskDataset` 和任务构造，支持 `reconstruct`、`separate_vocals`、`separate_accompaniment`、`super_resolution`、`mono_to_stereo`。
- `training_base.py` 只保留当前训练 wrapper 需要的 finite check、grad norm、EMA 更新、导出和 pseudo-Huber loss，不集成完整 `scripts.music2latent`。

## Separation 预处理

`scripts/data/prepare_separation.py` 使用本地 Kimberley Mel-Band RoFormer vocal model checkpoint/config 直接做 RoFormer 推理。实现参考 Kimberley 官方 `inference.py`：模型只输出 `vocals`，`accompaniment.mp3` 由 `mixture - vocals` 生成，不再读取 `other/Instrumental` stem。

- sample rate: `44100`
- channels: `2`
- model output stem: `vocals`
- saved stems: `mixture`、`vocals`、`accompaniment`
- 默认 inference overlap: `2`

模型文件默认从这里读取：

```text
checkpoints/mel-band-roformer-vocal-model/config_vocals_mel_band_roformer.yaml
checkpoints/mel-band-roformer-vocal-model/MelBandRoformer.ckpt
```

输出目录严格保留 `--source-root` 下的原始相对路径，并把原音频文件名去掉后缀作为条目目录。例如 `NETEASE_SPIDER/audio/20260520/1001230.mp3` 会输出到：

```text
NETEASE_SPIDER_SEPERATION_NEW/audio/20260520/1001230/
```

每首歌固定写入：

- `mixture.mp3`
- `vocals.mp3`
- `accompaniment.mp3`
- `metadata.json`

示例：

```sh
cd /inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland
/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python \
  -m scripts.data.prepare_separation \
  --input-dir /inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER/audio \
  --source-root /inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER \
  --output-root /inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER_SEPERATION_NEW \
  --device cuda:0 \
  --num-overlap 2 \
  --chunk-batch-size 1 \
  --save-workers 2 \
  --max-pending-writes 4
```

已有完整 `metadata.json(status=done)` 和三份 stem mp3 时自动跳过；加 `--overwrite` 可强制重算。小样本测试可加 `--max-files N`，扫描会在达到数量后短路。

`--save-workers` 会把 MP3 编码/写盘放到后台线程，让下一首 RoFormer 推理和上一首 MP3 写入重叠。`--max-pending-writes` 控制每个推理 worker 最多缓存多少个待保存结果；当前单卡建议先用 `--save-workers 2 --max-pending-writes 4`。

`--chunk-batch-size` 控制一首歌内部同时送进 RoFormer 的 chunk 数。本机 2 个 NETEASE_SPIDER 样本上，`2` 比默认 `1` 略慢，当前推荐保持 `1`。如果源文件已经是 44100 Hz、2 声道 mp3，脚本会直接复制源文件作为 `mixture.mp3`，只编码 `vocals.mp3` 和 `accompaniment.mp3`。

使用 `--devices` 多 worker 时，父进程终端显示总文件级进度条 `files`，并在 postfix 里显示 `done/skipped/skiplong/errors`；每个 worker 的详细 stdout/stderr 写到 `${output-root}/_logs/prepare_separation/worker_*.log`。worker 按文件懒检查终止状态，不会在首个 pending 文件处理前对整份 manifest 做全量 pending/skipped 预扫。

## 数据集接法

`scripts.data.datasets.PreparedSeparationDataset` 直接读取 `prepare_separation.py` 产物目录，并返回离线分离结果作为 payload：

- `audio`
- `mixture`
- `vocals`
- `accompaniment`

首次构造时会扫描 prepared 目录中的 `mixture.mp3`，生成 `${stems_root}/index.list`；之后训练启动直接读取该索引。`MosslandTaskDataset` 会先抽样任务，再通过 `get_item_for_task()` 让 dataset 按需解码 stem：非分离任务只读 `mixture.mp3`，`separate_vocals` 只额外读 `vocals.mp3`，`separate_accompaniment` 只额外读 `accompaniment.mp3`。分离任务只接受 prepared 数据中的显式 `mixture` 和目标 stem，不做其他 stem fallback。

`super_resolution.low_sample_rate` 可以是单个整数、显式 rate 列表，或 `[min, max]` 范围；当前实验配置使用 `[8000, 40000]`，会从区间内固定 audio super-resolution bucket（8000、11025、12000、16000、22050、24000、32000、40000）采样，而不是采任意整数。任意整数 rate 可能与 44100 互质，导致 torchaudio sinc resample kernel 极大并阻塞 DataLoader worker。audio super-resolution 降质使用缓存的 `torchaudio.transforms.Resample` 做降采样再升采样，保留 resample 的 sinc lowpass/anti-alias 行为；长度对齐只做裁剪或重复尾部 sample；audio super-resolution、mono-to-stereo 和分离任务都只通过 5 个 `task_id` 区分。

Hydra 训练入口：

```sh
python scripts/train.py experiment=mossland-codec
```

## 参数入口

训练配置在 `scripts/configs/experiment/mossland-codec.yaml` 的 `model:` 节显式传入 `MosslandCodecTransformer` 构造参数，包括采样率、STFT hop、压缩倍率、频谱长度、sigma 范围、latent 数量和 frontend 结构。任务列表、权重和 `low_sample_rate` 只配置在 `MosslandTaskDataset`；wrapper 只保留训练循环、优化器和 loss 相关参数。

推理入口示例：

```python
from importlib import import_module

EncoderDecoder = import_module("scripts.mossland-codec.inference").EncoderDecoder
codec = EncoderDecoder(
    load_path_inference="/path/to/checkpoint.ckpt",
    model_kwargs={"sample_rate": 48000, "hop": 1024, "fac": 2, "spec_length": 32},
)
```
