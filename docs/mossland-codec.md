# Mossland Codec 说明

本文件记录 `scripts/mossland-codec` 的当前架构、训练入口和 separation 数据准备方式。

## 设计决策

- `scripts/codicodec/` 和 `scripts/configs/experiment/codicodec.yaml` 保留，作为独立 CoDiCodec 参考实现。
- `scripts/mossland-codec/` 是 Mossland codec 唯一实现目录，使用 hyphen 路径，不再创建 `scripts/mossland_codec/`。
- `scripts/mossland-codec/` 不 import `scripts.codicodec`。需要的模型、音频表示、训练工具和推理辅助代码已经集成在本目录。
- `scripts/mossland-codec/hparams.py` 和 `scripts/mossland-codec/hparams_inference.py` 已删除。模型、音频表示、噪声调度和推理默认值不再从全局 hparams 模块读取，统一通过 `MosslandCodecTransformer(...)` 构造参数、`EncoderDecoder(model_kwargs=...)` 或 Hydra `model:` 配置显式传入。
- 模型类是 `scripts.mossland-codec.models.MosslandCodecTransformer`。decoder conditioning 只使用 sigma embedding 加上 5 个任务的 `task_embedding`；模型也负责 `prepare_audio_batch()` 和 `generate_waveform()` 这类与模型输入长度、声道和推理路径强相关的逻辑。
- wrapper 类是 `scripts.mossland-codec.wrapper.MosslandCodecTrainingWrapper`，负责消费已经构造好的任务 payload、计算 consistency loss、维护 EMA、optimizer/scheduler 和 demo 输出；不再抽样任务或构造 `src/target`。
- demo callback 在生成 demo 前后都会执行 `gc.collect()`、`torch.cuda.synchronize()` 和 `torch.cuda.empty_cache()`，避免 demo 推理与训练 step 的缓存显存重叠导致 OOM；每个样本会分别保存 `quantized` 与 `continuous` 两份 `src/target/generated` 对比音频，分别对应 `dont_quantize=False` 和 `dont_quantize=True`。
- 不集成完整 `scripts.music2latent`；只保留当前 wrapper 需要的 finite check、grad norm、EMA 更新、导出和 pseudo-Huber loss。

## 任务

`scripts.mossland-codec.tasks.TASK_NAMES` 当前包含：

- `reconstruct`
- `separate_vocals`
- `separate_accompaniment`
- `super_resolution`
- `mono_to_stereo`

`MosslandTaskDataset` 是唯一的训练任务抽样入口，把普通音频或 prepared stem payload 适配为 `src/target/task_id`。`MosslandTaskBatch` 负责标准 payload 的 `to_payload/from_payload` 转换，wrapper 只接受这个 schema。audio super-resolution 的具体 downsample rate 和 mono/stereo channel mode 不再作为额外条件传给模型。

## Separation 预处理

新增脚本：

```sh
python -m scripts.data.prepare_separation \
  --input-dir /path/to/music \
  --source-root /path/to/music \
  --output-root /path/to/mossland_separation
```

脚本按 KimberleyJensen/Mel-Band-RoFormer-Vocal-Model 的官方推理逻辑准备离线分离结果：模型只直接预测 `vocals`，`accompaniment.mp3` 由 `mixture - vocals` 生成。外部参考：

- https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/main/inference.py
- https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/main/configs/config_vocals_mel_band_roformer.yaml
- https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt

模型文件放在本地大文件目录，不提交到 git：

```text
checkpoints/mel-band-roformer-vocal-model/config_vocals_mel_band_roformer.yaml
checkpoints/mel-band-roformer-vocal-model/MelBandRoformer.ckpt
```

该配置的关键约定是 `model.sample_rate=44100`、`model.stereo=true`、`model.num_stems=1`、`training.target_instrument=vocals`、`training.instruments=[vocals, other]`、`inference.num_overlap=2`。`prepare_separation.py` 不内置 config 字典，默认必须能读取本地 `config_vocals_mel_band_roformer.yaml`。`_select_vocals()` 只接受模型输出里的精确 key `vocals`，避免旧模型字段兜底掩盖配置错误。

输出目录严格保留 `--source-root` 下的原始相对路径，并把原音频文件名去掉后缀作为条目目录。例如 `NETEASE_SPIDER/audio/20260520/1001230.mp3` 会输出到：

```text
NETEASE_SPIDER_SEPERATION_NEW/audio/20260520/1001230/
```

每个条目目录固定写入：

- `mixture.mp3`
- `vocals.mp3`
- `accompaniment.mp3`
- `metadata.json`

当前 `py_env` 已安装 `beartype` 和 `rotary-embedding-torch`，可以直接运行本仓库脚本。当前 Notebook 只有一张 RTX 4090，分解 `NETEASE_SPIDER` 的推荐命令：

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
  --max-duration-seconds 600 \
  --save-workers 2 \
  --max-pending-writes 4
```

脚本默认跳过已经有 `metadata.json(status=done)`、`mixture.mp3`、`vocals.mp3`、`accompaniment.mp3` 的条目；需要重跑时加 `--overwrite`。小样本测试可加 `--max-files N`，扫描会在达到数量后短路，不会枚举完整 `audio/` 目录。

`--save-workers` 会把 MP3 编码/写盘放到后台线程，让下一首 RoFormer 推理和上一首 MP3 写入重叠，缓解 GPU 因同步保存而空闲。`--max-pending-writes` 控制每个推理 worker 最多缓存多少个待保存结果；值越大越容易覆盖写盘延迟，但内存占用也越高。当前单卡建议先用 `--save-workers 2 --max-pending-writes 4`。

`--chunk-batch-size` 控制一首歌内部同时送进 RoFormer 的 chunk 数。它可以减少小 kernel 调度次数，但会增加显存和 CPU 时间；2026-06-08 在本机 2 个 NETEASE_SPIDER 样本上，`chunk_batch_size=2` 比 `1` 略慢，因此默认和当前推荐仍保持 `1`，后续只有在更长音频或不同 GPU 上重新 benchmark 后再调大。

如果源文件本身是与模型输入一致的 mp3（当前为 44100 Hz、2 声道），脚本会直接复制源文件作为 `mixture.mp3`，只编码 `vocals.mp3` 和 `accompaniment.mp3`。这样避免原音频二次 MP3 编码，也减少一份写盘 CPU 开销；不匹配时会自动回退到原来的 mixture 重编码路径。

脚本默认只处理短于 10 分钟的音频：`--max-duration-seconds 600` 会在完整加载音频和 RoFormer 推理前探测时长，优先使用 `ffprobe`，再回退到 `torchaudio.info()`；时长大于等于阈值的条目写 `metadata.json(status=skiplong, reason=max_duration_exceeded)`，并在 `progress.jsonl` 中记录 `status=skiplong`。普通已有完成产物仍记录为 `skipped`，两者在父进程进度条中分开统计。需要关闭时长限制时传 `--max-duration-seconds 0`。

使用 `--devices` 多 worker 时，父进程终端显示总文件级进度条 `files`，并在 postfix 里显示 `done/skipped/skiplong/errors`；每个 worker 的详细 stdout/stderr 写到 `${output-root}/_logs/prepare_separation/worker_*.log`。这只适合同一台机器内的多卡 launcher。

多机多卡不要在每台机器上对同一份 manifest 直接使用 `--devices`，因为当前 launcher 只按本机 `len(devices)` 生成 shard，会导致不同机器重复处理同一组 shard。多机推荐每张 GPU 启一个独立进程，所有进程共用同一份 `--files-list`、`--source-root` 和 `--output-root`，并设置全局 `--num-shards=<总GPU数>`、`--shard-id=<全局GPU序号>`、`--device cuda:<本机GPU序号>`。例如 4 台机器、每台 8 卡时，总 shard 数是 32；第 3 台机器（`NODE_RANK=2`）的本机第 5 张卡（`LOCAL_GPU=5`）使用 `--num-shards 32 --shard-id 21 --device cuda:5`。多机运行时建议给每个进程单独的 `--progress-file`，避免共享文件系统上的多进程追加写互相干扰；全局进度以 prepared 输出目录中的 `done/skiplong/error` metadata 为准。

`process_files()` 不再在启动时对整份 manifest 调 `filter_pending_files()` 做全量终止状态预扫。worker 会按 manifest 顺序逐条检查：已有完整 `done` 产物立即写 `skipped` 进度；已有 `metadata(status=error)` 会先按当前 `--max-duration-seconds` 用同一套时长探测重判，若超长则改写为 `skiplong` 并直接跳过，否则才计入 `error`；第一个真正 pending 的文件会立即进入时长检查，超长直接 `skiplong`，短音频才进入 RoFormer 推理，模型实例也只在首个需要推理的文件前懒加载。这个约束用于避免 624 万级 manifest 在首个文件处理前长时间停在 `files 0%`。

launcher 会增强 native 崩溃场景的稳健性：worker 在处理每个 pending 文件前写 `progress.jsonl(status=started)`。如果子进程被 SIGSEGV 等 native 错误打死（例如 launcher 看到 `rc=-11`），父进程会用该 worker 最后的 `started` 文件写 `metadata.json(status=error, reason=worker_crash)`，然后重启同一个 shard。重启后 `metadata(status=error)`、`metadata(status=skiplong)` 和完整 `done` 条目都会被视为终止状态，除非加 `--overwrite`，不会反复撞同一个文件。`--worker-restarts` 控制每个 worker 的最多自动重启次数，默认 100。

## Separation 数据集

`scripts.data.datasets.PreparedSeparationDataset` 直接读取 `prepare_separation.py` 已生成的离线目录，不再需要先构造原始 `SampleDataset`。它读取 `mixture.mp3/vocals.mp3/accompaniment.mp3` 成套存在的 prepared 条目：

- `mixture`
- `vocals`
- `accompaniment`

首次构造时会用 `fast_scandir` 扫描 prepared 目录的 `audio/` 子树，找到 `mixture.mp3` 且同目录包含 `vocals.mp3/accompaniment.mp3` 的条目，并写入 `${dirs[0]}/index.list`。之后如果 `index.list` 存在，训练启动直接读取它，不再依赖 `prepare_separation.py` 的 progress 日志。每个样本会使用同一个 sample offset 裁切已解码的 stem，并返回 `audio=mixture`，所以 `reconstruct`、`super_resolution`、`mono_to_stereo` 仍可直接使用 mixture。

`PreparedSeparationDataset.length` 可把 dataset 暴露给 DataLoader 的逻辑长度固定住；为 `None` 时使用真实 `len(item_dirs)`。取样时仍会对当前真实 `item_dirs` 数量取余，因此固定 `length` 不会复制样本，只是提供稳定 epoch 长度。当前主配置设为 `1000000`，适合分离预处理边生成、训练边读取的场景。

`MosslandCodecTrainingWrapper.index_data_every_step` 可在训练过程中周期性重新索引 prepared 目录；为 `None` 时不重新索引。wrapper 在 `on_train_batch_end()` 按 `trainer.global_step` 判断是否触发，触发时从 `trainer.datamodule` 找到带 `rebuild_index()` 的底层 dataset 并调用它。`PreparedSeparationDataset.rebuild_index()` 会重新扫描 prepared 目录、重写 `index.list`，并直接替换当前 dataset 实例的 `self.item_dirs`。

`MosslandTaskDataset` 会先抽样任务，再调用 dataset 的任务感知读取接口，避免非分离任务白白解码 stem。单 crop 路径使用 `get_item_for_task(index, task_id)`：非分离任务只读 `mixture.mp3`；`separate_vocals` 只读 `mixture.mp3` 和 `vocals.mp3`；`separate_accompaniment` 只读 `mixture.mp3` 和 `accompaniment.mp3`。

`PreparedSeparationDataset.crops_per_file` 用于一次读取同一个 prepared item 的 mp3 stem 后，在内存里裁出多个训练片段并 stack 成 `[crops_per_file, channels, time]`。当 `crops_per_file > 1` 且 dataset 提供 `get_item_for_tasks()` 时，`MosslandTaskDataset` 会为每个 crop 独立随机采样任务，把任务列表传给 prepared dataset；prepared dataset 只解码这些任务需要的 stem 并集，例如一组 crop 中同时有 vocals/accompaniment 分离任务时，同一个 item 只会各读一次 `mixture.mp3/vocals.mp3/accompaniment.mp3`。DataLoader collate 后的 `[train_batch_size, crops_per_file, channels, time]` 会由 `MosslandTaskBatch.from_payload()` 展平成有效训练 batch，并按 batch-major 顺序保留每个 crop 的 `task_id`。当前主训练配置使用 `crops_per_file=16`、`train_batch_size=1`，有效 batch 为 16。

`PreparedSeparationDataset.max_duration_seconds` 用于在读取 mp3 前跳过过长样本。当前主配置和 small 配置都设为 `300`：dataset 会在选中单条 item 后、真正解码 mp3 前，用 `ffprobe` 探测 `mixture.mp3` 容器时长；大于等于 5 分钟的条目会从当前 worker 的 item list 中移除并重采样。过滤不在训练启动时全量扫描 index，避免 10 万级 prepared item 因逐条 ffprobe 导致启动变慢。ffprobe 失败时保留样本，避免因为探测工具异常误删数据。

多卡训练时 DataLoader 首批预取会按 `devices * num_workers * prefetch_factor * train_batch_size` 放大 prepared item 数量；每个 item 平均还会按随机任务解码约 2 到 3 个 stem mp3。2026-06-09 本机排查发现，旧配置 `num_workers=10,prefetch_factor=4` 在 8 卡、`train_batch_size=4` 时会在首批前排队约 1280 个 item、三千级 mp3 解码请求，容易让共享存储长时间卡在 `Epoch 0 0/N`。当前主配置为 `num_workers=6,prefetch_factor=2`，需要继续观察共享存储压力。

## 训练入口

默认多任务训练入口：

```sh
python scripts/train.py experiment=mossland-codec
```

`scripts/configs/experiment/mossland-codec.yaml` 的 `model:` 节是模型超参数的主入口，包含 `sample_rate`、`hop`、`fac`、`spec_length`、`sigma_min`、`sigma_max`、`num_latents`、frontend 层数等原先容易散落在 hparams 文件中的参数。训练 wrapper 只从传入的 `model` 实例读取需要保持一致的音频和噪声调度参数。

当前训练配置把模型容量调到约 494M 参数：`dim=768`、`head_dim=128`、`num_layers=22`、`num_layers_encoder=22`、`cond_channels=768`、`frontend_multipliers_list=[1,2,4,12]`。压缩行为保持不变：`hop=1024`、`fac=2`、`spec_length=32`、`num_latents=128`、`fsq_levels=[11,11,11,11]` 和 `frontend_freq_downsample_list=[0,1,0]` 不变；实测实例化后 `data_length=512`、`freq_dim=64`、`time_dim=8`、`downsample_ratio=64`。

`scripts/mossland-codec/transformer_layers.py` 的 attention 通过 PyTorch `scaled_dot_product_attention` 执行。CUDA bf16/fp16 且无显式 mask 时会用 `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` 强制 flash backend；CPU、不支持 flash 或显式 tensor mask 时回退普通 SDPA。decoder 原先的 `[2*block, 2*block]` float block mask 会让 SDPA 退回非 flash，现在改为 `block_causal_attention_mask(block_size)` 轻量 spec，`MultiHeadAttention` 把它拆成两次无 mask attention：左半只 attend 左半，右半 attend 全部，语义等价但可走 flash。

当前 `scripts/configs/experiment/mossland-codec.yaml` 已直接使用 `PreparedSeparationDataset` 指向：

```text
/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER_SEPERATION_NEW
```

训练读取阶段的 `source_root` 已从 `PreparedSeparationDataset` 和实验配置中删除：dataset 扫描和加载只依赖 `dirs`、`index.list` 和 prepared item 目录内的 stem mp3。`source_root` 只保留在 `prepare_separation.py` 生成阶段，用于把原始音频映射到 prepared 输出相对路径。同时 `MosslandTaskDataset` 的 `active_tasks` 已启用 `separate_vocals` 与 `separate_accompaniment`。wrapper 配置不再包含 `active_tasks`、`task_weights` 或 `low_sample_rate`，避免训练侧重复采样任务。`Experiment_Dataset` 对很小的正数 `val_split` 会至少保留 1 条验证样本，避免 prepared 数据尚未全部生成时验证集长度为 0。

`super_resolution` 的 `low_sample_rate` 可以是单个整数、显式 rate 列表，或 `[min, max]` 范围。当前配置使用 `[8000, 40000]`，每次构造该任务时会从区间内固定 audio super-resolution bucket（8000、11025、12000、16000、22050、24000、32000、40000）采样一个低采样率来生成降质 `src`，但 payload 仍只有 `task_id="super_resolution"`，模型不接收具体 rate 条件。不要在训练热路径中采任意整数 rate：例如与 44100 近似互质的 rate 会让 torchaudio sinc resample kernel 极大，阻塞 DataLoader worker。audio super-resolution 降质使用缓存的 `torchaudio.transforms.Resample` 先降到低采样率再升回训练采样率，依赖 sinc lowpass/anti-alias 过滤超过低采样率 Nyquist 的频率；resample 后如有长度误差，只裁剪或重复尾部 sample，不再对整段音频做线性插值。分离任务只接受 `PreparedSeparationDataset` 提供的显式 `mixture` 和目标 stem，不再根据 `vocals/accompaniment` 或其他 stem 兜底合成 mixture。

## 推理入口

推理辅助类 `scripts.mossland-codec.inference.EncoderDecoder` 通过 `scripts.factory.load_model()` 读取模型。推荐传 checkpoint 目录，目录内应包含 `config.yaml` 和 `checkpoint.ckpt`：

```python
from importlib import import_module

EncoderDecoder = import_module("scripts.mossland-codec.inference").EncoderDecoder
codec = EncoderDecoder(
    load_path_inference="/path/to/ckpt_dir",
)
latents = codec.encode("/path/to/audio.wav")
audio = codec.decode(latents, task_id="reconstruct")
```

`decode()`、`decode_next()` 和底层分块 decode 默认 `task_id="reconstruct"`，也可传 `separate_vocals`、`separate_accompaniment`、`super_resolution` 或 `mono_to_stereo`。如果未显式传入 `max_batch_size_encode`、`max_batch_size_decode` 或 `sigma_rescale`，推理代码会读取模型实例上的同名属性。

## 推理相关参数

- `default_time_prompt` 是 autoregressive/live decode 的默认历史提示噪声时间值。`time_prompt=None` 时，`decode_autoregressive()` 和 `decode_autoregressive_step()` 使用该值，经 `get_sigma_continuous()` 映射成给上一段 clean 输出重新加噪的 `sigma_prompt`；当前默认 `0.4` 在 `sigma_min=0.002`、`sigma_max=80.0`、`rho=7.0` 下约等于 `sigma=0.965`。
- `sigma_rescale` 是连续 latent 的对外缩放系数。encode 返回连续 latent 前会做 `atanh(latent) / sigma_rescale`；decode 连续 latent 时会反向做 `tanh(latent * sigma_rescale)` 再送回模型内部 latent 形状。它不是 STFT 幅度归一化里的同名局部参数。
