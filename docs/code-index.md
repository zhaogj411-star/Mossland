# 代码索引

Mossland 当前包含 agent harness 和音乐多任务 codec 训练代码。

## 仓库根部

- `AGENTS.md`：仓库代理规则。
- `.codex/hooks.json`：Codex 生命周期 hook 配置；命令指向 `agent-code/scripts/codex-hook.sh`。
- `agent-code/`：完全由机器管理的 agent harness 代码根目录。
- `agent-code/scripts/agent/`：显式 preflight、impact 和验证命令。
- `agent-code/scripts/codex-hook.sh`：生命周期提醒 hook。
- `agent-code/tests/agent/`：agent harness 测试。
- `bash/prepare_seperation/node0.sh` 到 `node3.sh`：4 台机器多机多卡 separation 预处理启动脚本；每台机器 8 卡，总 `--num-shards 32`，每张 GPU 一个独立进程，使用全局 `--shard-id`，日志写入 `NETEASE_SPIDER_SEPERATION_NEW/_logs/prepare_separation_multinode/`。
- `scripts/train.py`：Hydra/Lightning 训练入口。
- `scripts/codicodec/`：保留的 CoDiCodec 独立参考实现和原始实验入口依赖代码；`scripts/mossland-codec/` 不 import 它。
- `scripts/mossland-codec/`：Mossland 多任务 codec 自包含实现，包含 `MosslandCodecTransformer`、音频表示、consistency decoder、训练 wrapper、任务适配层和最小训练基类；模型负责 `prepare_audio_batch()` 与 `generate_waveform()`，decoder conditioning 只使用 sigma embedding 加 5 个 `task_id` embedding；wrapper 只消费 `src/target/task_id` 标准任务 payload 并执行训练循环，不做跨样本 `random_mix`；demo callback 每个样本同时保存 quantized 与 continuous 两份 `src/target/generated` 对比音频；不集成完整 `scripts.music2latent`，不保留 `hparams.py` 或 `hparams_inference.py`。
- `scripts/mossland-codec/transformer_layers.py`：Transformer attention 层。CUDA bf16/fp16 无显式 tensor mask 时强制 PyTorch SDPA flash backend；decoder 的 block causal mask 用两段无 mask attention 保持语义并避免 flash 退化。
- `scripts/configs/experiment/mossland-codec.yaml`：Mossland 多任务 codec 训练配置，命令为 `python scripts/train.py experiment=mossland-codec`；`model:` 节显式传入模型、音频表示和噪声调度参数，当前容量约 494M 参数且保持压缩相关 `hop/fac/spec_length/num_latents/fsq_levels/frontend_freq_downsample_list` 不变；dataset 直接使用 `PreparedSeparationDataset` 读取 `NETEASE_SPIDER_SEPERATION_NEW` prepared folder，当前 `crops_per_file=16`、`train_batch_size=1`，一次 mp3 解码提供 16 个 crop、每个 crop 独立随机任务，展平后有效 batch 为 16；`length=1000000` 固定逻辑 epoch 长度，真实条目用当前 `item_dirs` 取余；wrapper 的 `index_data_every_step=null` 表示训练中不重建索引，改成正整数后会在 `on_train_batch_end()` 按 `trainer.global_step` 调用底层 dataset 的 `rebuild_index()`；`max_duration_seconds=300` 会在选中单条 item 后用 `ffprobe` 在 mp3 decode 前 lazy 过滤大于等于 5 分钟的条目；当前 `num_workers=6,prefetch_factor=2`，需要继续观察多卡首批预取的共享存储压力；任务列表、权重和 `low_sample_rate` 只配置在 `MosslandTaskDataset`，wrapper 不再重复维护这些参数；`super_resolution.low_sample_rate=[8000,40000]` 表示从固定 fast audio SR bucket 采样，不在 DataLoader 热路径采任意整数 rate。
- `scripts/data/`：音频数据集和 DataModule 工具。
- `scripts/data/prepare_separation.py`：按本地 Kimberley Mel-Band RoFormer vocal model config/ckpt 离线生成 `mixture.mp3/vocals.mp3/accompaniment.mp3` 分离结果；默认读取 `checkpoints/mel-band-roformer-vocal-model/config_vocals_mel_band_roformer.yaml` 和 `MelBandRoformer.ckpt`，模型只输出精确 key `vocals`，`accompaniment` 由 `mixture - vocals` 生成；输出目录严格保留 `--source-root` 下的相对路径；支持 `--device cuda:0` 单卡、`--devices 0,1,...` 多卡 worker、父进程文件级聚合进度条、`--max-files` 扫描短路、按文件懒检查 pending/skipped 以避免大 manifest 启动前全量预扫、已完成条目跳过、默认 10 分钟以上音频 `skiplong` 跳过、旧 `metadata(status=error)` 超长重判并改写 `skiplong`、native 崩溃后按 `started` 当前文件标记 `worker_crash` 并重启 worker、异步 MP3 保存、`--chunk-batch-size` demix chunk batching，以及匹配源 mp3 直接复制为 `mixture.mp3`。
- `scripts.data.datasets.PreparedSeparationDataset`：直接扫描 `prepare_separation.py` 产物目录并写入/读取 `index.list`，支持固定对外 `length`、用真实 `item_dirs` 取余读取，以及通过 `rebuild_index()` 重新扫描并替换 `self.item_dirs`；可用 `max_duration_seconds` 在单条 item 被选中后通过 `ffprobe` 按 `mixture.mp3` 容器时长 lazy 过滤长音频，按单任务或每 crop 任务列表只解码需要的 `mixture/vocals/accompaniment` stem 并集，供 `MosslandTaskDataset` 多任务训练使用；旧的包装式 separation stem dataset 已删除。
- `scripts/third_party/mel_band_roformer/`：从同事使用的 Mel-Band-Roformer-Vocal-Model 复制的最小模型实现，用于 `prepare_separation.py` 本地直推。
- `scripts/trainer_utils/`：Hydra/Lightning logger、callback 和通用训练工具。
- `tests/mossland_codec/`、`tests/test_prepare_separation.py`、`tests/test_separation_dataset.py`：Mossland codec、任务适配和 separation 预处理/读取的轻量单元测试。
- `docs/mossland-codec.md`：Mossland codec 架构、separation 预处理和数据集接法说明。
- `docs/`：持久项目记忆。

新增、移动或废弃代码、入口点、配置、测试或资源边界时，更新本文件。
