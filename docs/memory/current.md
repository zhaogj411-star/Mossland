# 当前上下文

本文件是新 Codex 会话的短活跃上下文。保持简洁。

## 当前工作

- Mossland 代理基础设施已从 `moss-training-framework` 的通用规则迁移而来。
- 仓库文档和 agent harness 面向人阅读的输出已统一改为中文。
- `scripts/mossland-codec` 第一版已改为自包含多任务 codec，实现 `MosslandCodecTransformer`、训练 wrapper、任务数据适配层、Hydra 配置、separation 预处理和 prepared separation dataset。
- `scripts/mossland-codec` 已移除 `hparams.py` 和 `hparams_inference.py`；模型、音频表示、噪声调度和推理默认项通过 `MosslandCodecTransformer(...)`、Hydra `model:` 或通过 `scripts.factory.load_model()` 从 `config.yaml/checkpoint.ckpt` 读取。
- 已完成 `scripts/mossland-codec/tasks.py` 5 个任务的评估指标调研，报告在 `docs/evaluation-metrics.md`，论文 PDF 在 `docs/papers/`；`mono_to_stereo` 按生成式 stereo rendering/upmix 任务处理，不把某个双声道 reference 当唯一真值。

## 稳定决策

- 使用 `docs/` 作为持久项目记忆根目录。
- 在 Mossland 专属代码出现前，保持导入的 agent harness 通用。
- 除非内容已成为当前 Mossland 需求、决策或工作流，不复制训练框架的长篇文档。
- 面向人阅读的文档、agent harness 说明、hook 提醒、脚本输出和未来 harness 设定默认使用中文；命令名、路径、scope、事件名、JSON key 和代码标识保留英文接口。
- 完全由机器管理的 harness 代码集中在 `agent-code/`；`.codex/hooks.json` 仍留在 `.codex/` 作为 Codex 配置入口。
- `scripts/mossland-codec` 只保留一个 hyphen 目录；不要再创建 `scripts/mossland_codec`。
- 模型入口统一命名为 `MosslandCodecTransformer`，因为核心实现是 Transformer/Transformer_Diffusion；不要再恢复旧的 `UNet` 命名。
- `scripts/codicodec/` 和 `scripts/configs/experiment/codicodec.yaml` 保留为独立参考实现；`scripts/mossland-codec/` 不 import `scripts.codicodec`。
- Mossland 不集成完整 `scripts.music2latent`，只把当前 wrapper 需要的 `pseudo_huber_loss` 和训练基类逻辑重构进 `scripts/mossland-codec/training_base.py`。
- 不再新增或恢复 `scripts/mossland-codec/hparams.py`、`scripts/mossland-codec/hparams_inference.py`；新增参数应进入模型构造签名或 Hydra `model:` 配置；推理 `EncoderDecoder` 默认通过 `scripts.factory.load_model()` 读取 checkpoint 目录里的 `config.yaml/checkpoint.ckpt`。
- `prepare_separation.py` 默认使用本地 `checkpoints/mel-band-roformer-vocal-model/`，包含 `config_vocals_mel_band_roformer.yaml` 和 `MelBandRoformer.ckpt`；`checkpoints/` 是本地大文件目录，不提交。代码不内置网络 URL 或 config 字典，`load_model_config(None)` 必须读取本地默认 config 文件。
- `prepare_separation.py` 参考 KimberleyJensen/Mel-Band-Roformer-Vocal-Model 官方 `inference.py`：模型只直接输出 `training.target_instrument=vocals`，`_select_vocals()` 只接受精确 key `vocals`；`accompaniment.mp3` 由 `mixture - vocals` 生成，不读取 `other/Instrumental` stem。
- `prepare_separation.py` 默认写 `mixture.mp3`、`vocals.mp3`、`accompaniment.mp3` 和 `metadata.json`；跳过逻辑只认三份 stem mp3 与 `metadata.json(status=done)`。不要再生成 wav stem。
- `prepare_separation.py` 的输出目录严格保留 `--source-root` 下的相对路径，例如 `NETEASE_SPIDER/audio/20260520/1001230.mp3` 输出到 `NETEASE_SPIDER_SEPERATION_NEW/audio/20260520/1001230/`，不要再使用 `audio__20260520__1001230` 扁平目录。
- 当前 Notebook 只有一张 RTX 4090；`NETEASE_SPIDER` 分解目标目录是 `/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/data/NETEASE_SPIDER_SEPERATION_NEW`。训练配置也指向这个 NEW prepared root。
- `prepare_separation.py` 支持 `--devices` 单机多卡 worker；父进程会聚合 worker 写入的 `progress.jsonl`，终端显示总文件进度 `files done/skipped/skiplong/errors / total`，不再显示 `workers 0/N`。多机多卡不要在每台机器对同一 manifest 直接用 `--devices`，应每张 GPU 起一个独立进程并用全局 `--num-shards/--shard-id` 分片。4 台机器每台 8 卡的启动脚本已写在 `bash/prepare_seperation/node0.sh` 到 `node3.sh`。当前机器只有一张 RTX 4090，实际应用 `--device cuda:0` 单 worker；`--max-files N` 会扫描短路，适合本地小样本测试。
- `prepare_separation.py` 的 `process_files()` 不能在启动时对整份 manifest 全量预扫 pending/skipped。624 万级 manifest 会长时间停在 `files 0%`，worker 日志栈在 `filter_pending_files -> separation_status -> safe_stem_id`。当前实现按文件逐条检查终止状态，第一个 pending 文件立即进入时长检查和 RoFormer 推理，模型只在首个需要推理的文件前懒加载。
- `prepare_separation.py` 支持异步 MP3 保存：`--save-workers` 控制后台保存线程数，`--max-pending-writes` 控制最多缓存多少个待保存结果。当前单卡建议 `--save-workers 2 --max-pending-writes 4`，用于让 RoFormer 推理和上一首 MP3 写入重叠，缓解 GPU/CPU 都低的同步等待问题。
- `prepare_separation.py` 支持 `--chunk-batch-size` 控制 RoFormer 单次前向的 chunk 数；2026-06-08 本机 2 个 NETEASE_SPIDER 样本上 `2` 比默认 `1` 略慢，当前推荐保持 `--chunk-batch-size 1`，不要盲目调大。
- `prepare_separation.py` 写 `mixture.mp3` 时，如果源文件已经是 44100 Hz、2 声道 mp3，会直接复制源 mp3，避免原音频二次编码；不匹配时仍重编码 mixture。
- 2026-06-09 旧 `NETEASE_SPIDER_SEPERATION` 抽样 8 个 done 条目，每个只解码前 15 秒：源 MP3 与保存的 `mixture.mp3` 均为 44100 Hz 双声道，L/R 声道都不是完全相同或 `1e-7` 内近似相同；8/8 的 `mixture.mp3` 与源 MP3 字节完全相同，这是直接复制优化预期行为，不代表 L/R 被复制成同一声道。
- `prepare_separation.py` 默认只处理短于 10 分钟的音频：`--max-duration-seconds 600` 会在完整加载和 RoFormer 推理前探测时长，优先使用 `ffprobe`，再回退到 `torchaudio.info()`；超过或等于阈值的条目写 `metadata.json(status=skiplong, reason=max_duration_exceeded)` 和 `progress.jsonl(status=skiplong)`。传 `--max-duration-seconds 0` 可关闭限制。worker 重新扫到旧 `metadata(status=error)` 时会先按当前上限重判时长，超长则改写成 `skiplong` 并直接跳过，不进入 RoFormer。
- `prepare_separation.py` 多卡 launcher 会在 worker 处理 pending 文件前写 `progress.jsonl(status=started)`；若 worker native 崩溃（如 `rc=-11`），父进程按最后 `started` 文件写 `metadata.json(status=error, reason=worker_crash)` 并重启同一 shard。`--worker-restarts` 默认 100；重启后 `done/skiplong` 都是终止状态，旧 `error` 会先做时长重判，仍不超长才作为错误终止态跳过；除非 `--overwrite` 不会重复推理。
- `PreparedSeparationDataset` 使用 prepared root 下的 `index.list` 做持久索引；没有索引时用 `fast_scandir` 扫描 `audio/` 子树里的 `mixture.mp3`，筛出同目录包含 `vocals.mp3/accompaniment.mp3` 的条目并写入索引。它不再读取 `prepare_separation.py` 的 progress 日志。当前配置 `length=1000000` 固定逻辑 epoch 长度，真实读取时对当前 `item_dirs` 数量取余；wrapper 的 `index_data_every_step=null` 表示训练中不重建索引，改成正整数后 `MosslandCodecTrainingWrapper.on_train_batch_end()` 会按 `trainer.global_step` 调用底层 dataset 的 `rebuild_index()`，该方法会重新扫描并直接替换 `self.item_dirs`。当前配置 `max_duration_seconds=300`，dataset 只在选中单条 item 准备读取前用 `ffprobe` 探测 `mixture.mp3` 时长，若大于等于 5 分钟则从当前 worker 的 item list 跳过，不在训练启动时全量 probe index。
- `PreparedSeparationDataset.crops_per_file` 会一次加载同一个 prepared item 的 mp3 stem 后裁多个 crop；当前 `mossland-codec.yaml` 用 `crops_per_file=16`、`train_batch_size=1`，由 `MosslandTaskBatch.from_payload()` 把 `[1, 16, C, T]` 展平成有效 batch 16，减少每 step mp3 读取数。
- `MosslandTaskDataset` 会先抽样任务，再调用 dataset 的任务感知读取接口，让 `PreparedSeparationDataset` 按任务解码：单任务路径非分离任务只读 `mixture.mp3`，分离任务只额外读对应目标 stem；多 crop 路径会对每个 crop 独立随机任务，并通过 `get_item_for_tasks(index, task_ids)` 只解码任务列表需要的 stem 并集。分离任务只接受显式 `mixture` 和目标 stem，不再做 `mix`、stem 求和等兜底；旧的包装式 separation stem dataset 已删除。
- 多卡训练不要把 DataLoader 预取开太大。旧 `num_workers=10,prefetch_factor=4` 在 8 卡、`train_batch_size=4` 时首批前会排队约 1280 个 prepared items、三千级 mp3 解码请求，容易让共享存储长时间卡在 `Epoch 0 0/N`；当前主配置是 `num_workers=6,prefetch_factor=2`。
- `mossland-codec.yaml` 当前模型容量约 494M 参数：`dim=768`、`num_layers=22`、`num_layers_encoder=22`、`cond_channels=768`、`frontend_multipliers_list=[1,2,4,12]`。压缩行为相关 `hop=1024`、`fac=2`、`spec_length=32`、`num_latents=128`、`fsq_levels=[11,11,11,11]` 和 `frontend_freq_downsample_list=[0,1,0]` 保持不变。
- `scripts/mossland-codec/transformer_layers.py` 当前 attention 使用 PyTorch SDPA，并在 CUDA bf16/fp16 无显式 tensor mask 时用 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` 强制 flash backend。decoder 的 block causal mask 不再构造大 float mask，而是用两段无 mask attention 保持语义并避免 flash 退化。
- `MosslandTaskDataset` 是唯一的训练任务抽样和任务 batch 构造入口；`MosslandTaskBatch` 负责标准 payload 的 `src/target/task_id` 三字段 `to_payload/from_payload`，`MosslandCodecTrainingWrapper` 只消费这个 payload，不再维护 `active_tasks/task_weights/low_sample_rate`，也不再 fallback 构造任务。
- 当前 5 个 Mossland 任务 ID 是 `reconstruct`、`separate_vocals`、`separate_accompaniment`、`super_resolution`、`mono_to_stereo`；旧 `reconstruct_music` 和 `music_bandwidth_extension` 不再作为任务名使用。
- `MosslandCodecTransformer` 负责 `prepare_audio_batch()` 和 `generate_waveform()`；wrapper 训练 step 和 demo callback 直接调用当前模型或 EMA 模型的方法，不再保留 `_unpack_batch/_task_from_payload/_prepare_audio_batch` 这类转发函数。decoder conditioning 只使用 sigma embedding 加 5 个 `task_id` embedding。
- demo callback 必须在 demo 生成前后清理 CUDA cache，避免 demo 显存与训练显存重叠导致 OOM；每个 demo 样本会同时保存 `quantized` 与 `continuous` 两份 `src/target/generated` 对比音频，用于判断 FSQ quantized path 与 continuous path 的差异。
- `MosslandCodecTrainingWrapper` 不再做跨样本 `random_mix`，因为它会破坏 separation、mono/stereo 和 `super_resolution` 等任务的 `src/target` 对应关系；wrapper 配置中也不再保留 `random_mix_prob`。
- `super_resolution` 的 `low_sample_rate` 支持单个 int、显式 rate 列表或 `[min, max]` 范围；当前 `mossland-codec.yaml` 配置为 `[8000, 40000]`，会从区间内固定 bucket（8000、11025、12000、16000、22050、24000、32000、40000）采样，不采任意整数。audio super-resolution 降质用缓存的 `torchaudio.transforms.Resample` 做降采样再升采样，保留 sinc lowpass/anti-alias 行为；长度对齐只裁剪或重复尾部 sample，避免整段插值阻塞 DataLoader。mono-to-stereo 也不再写 `channel_mode`，只通过 `task_id="mono_to_stereo"` 区分。
- `separate_accompaniment` 的 target 来自 RoFormer prepared `accompaniment.mp3` teacher stem，不是人工干净伴奏。2026-06-09 run `2026-06-09_16-52-30` 的 demo target 与 src 余弦相似度约 `0.998`，抽样 prepared 条目也常见 `accompaniment` 接近 `mixture`；这表示 teacher stem 可能有人声残留或分离失败，后续需要保存 demo sidecar 元数据并考虑 separation 数据质量过滤。

## 下一步

- 未来新增或修改文档、hook 提醒、脚本输出和 harness 设定时，默认继续使用中文。
- 验证入口使用 `agent-code/scripts/agent/check.sh <scope>`。
- Mossland codec 训练入口使用 `python scripts/train.py experiment=mossland-codec`；当前直接按文件路径运行 `scripts/train.py` 时需要把仓库根目录放进 `PYTHONPATH`。
- 本机单卡连通性短跑命令和注意事项见 `docs/agent/commands.md` 的 “Mossland codec 训练短跑”。2026-06-09 已验证 1 step 可完成并写出 ckpt/demo。
- Mossland codec 架构和 separation 数据准备细节见 `docs/mossland-codec.md`。
- Mossland 多任务评估指标、论文来源和相关代码仓库见 `docs/evaluation-metrics.md`；论文索引见 `docs/papers/README.md`。
- `scripts/mossland-codec/inference.py` 的 `EncoderDecoder` 已改为通过 `scripts.factory.load_model()` 加载模型，并且 `decode()`、`decode_next()`、parallel/autoregressive 分块 decode 支持 `task_id`；`scripts/mossland-codec/infer.py` 是硬编码示例入口，使用同一 `EncoderDecoder` 打印连续 latent、离散 index/code 和生成音频形状。
- `2026-06-09_09-00-36` 训练 run 的 `latent/std` 后期两档切换已在 `docs/memory/progress.md` 记录：更像 FSQ latent 饱和/极端码使用风险，不是已证实的 NaN 式训练崩溃；后续优先补 pre-FSQ 与 code usage 监控。

## 阻塞项

- 无。
