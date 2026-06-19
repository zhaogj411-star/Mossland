# Music2Latent 官方复现状态

读取本文件用于判断 `scripts/configs/experiment/music2latent.yaml` 与 `tmp/music2latent-training (1)/music2latent-training` 官方训练代码的对齐程度。

## 当前结论

- `scripts/configs/experiment/music2latent.yaml` 已切为官方 public training code 复现口径：mono waveform、`hop=512/fac=4/data_length=64`、34304-sample crop、`data_channels=2`、`base_channels=64`、continuous latent only、RAdam、10k warmup、cosine 到 `final_lr=1e-6`、800k steps、`iters_per_epoch=10000` 等价逻辑长度 160000、EMA 0.9999、lognormal sigma 10000-bin weighted sampling、`base_step=0.1/end_exp=3`。当前 precision 按用户后续要求使用 `bf16-mixed`；官方原始代码是 fp16 autocast + GradScaler。
- 训练入口仍是本仓库 Hydra/Lightning，而不是官方自定义 `Trainer`。模型和 loss 语义已按官方对齐，但 checkpoint 保存、logger、DDP sampler 注入和 callback 由 Lightning 管理。
- 本地数据源不是官方示例配置中的 wav/flac 数据集路径；为了避免训练热路径解码音频，已从 NETEASE_SPIDER 索引预读取 10000 条 mono crop 到 `tmp/music2latent_official_pt_10k/`，训练 dataset 逻辑长度设为 160000 并对 10000 个物理 `.pt` 取模复用。本地原始索引主要是 mp3（8417605 条 mp3、23 条 flac），因此预读取阶段包含 mp3；训练时只读 `.pt` waveform。

## 关键文件

- 官方参考目录：`tmp/music2latent-training (1)/music2latent-training`。
- 本地训练入口：`scripts/configs/experiment/music2latent.yaml`。
- 本地实现：`scripts/music2latent/audio.py`、`scripts/music2latent/models.py`、`scripts/music2latent/wrapper.py`、`scripts/music2latent/data.py`。
- 预读取数据：`tmp/music2latent_official_pt_10k/{000000..009999}.pt`、`files.list`、`index.jsonl`。

## 长度与 STFT 约束

- 官方 64 个 STFT frame 对应 waveform 长度不是 `64*hop`，而是 `hop*64 + (fac-1)*hop = 34304`。
- `AudioProcessor(center_pad=false)` 使用官方 no-center-pad framing；`center_pad=true` 仅保留给旧实验兼容。
- 训练 wrapper 和 `Music2Latent.encode()` 都按 no-center-pad 公式对齐长度：`frames = (samples - fac*hop)//hop + 1`，然后把 frame 数对齐到 3 次时间 stride 的 8 倍。
- 已验证 batch `(2,34304)` 经 `to_representation_encoder()` 得到 `(2,2,1024,64)`。

## 验证记录

- `python -m py_compile scripts/music2latent/audio.py scripts/music2latent/data.py scripts/music2latent/models.py scripts/music2latent/wrapper.py` 通过。
- 预读取数据检查：10000 个 `.pt`，`files.list/index.jsonl` 均为 10000 行；每个 tensor shape/storage 都是 `34304`，目录约 `1.6G`。
- 单卡 1-step smoke 通过：`python -m scripts.train experiment=music2latent trainer.devices=1 trainer.strategy=auto trainer.max_steps=1 trainer.max_epochs=1 data.train_batch_size=1 data.num_workers=0 callbacks.demo_callback.demo_every=999999 logger.wandb=null trainer.enable_model_summary=false trainer.num_sanity_val_steps=0 extras.print_config=false`，得到有限 `loss/total=0.042`。

## 剩余差异

- 官方代码用自定义 loop、`torch.compile` 和 GradScaler；本地用 Lightning AMP 和 callback 系统。
- 官方训练期间每 epoch 做 FAD/test audio；本地默认 demo callback 保存 raw/EMA audio，FAD 仍需走仓库 eval pipeline 单独跑。
- 本地八卡配置每卡 batch 2，总 batch 16，等价官方 `batch_size=16`；dataset 逻辑长度 160000，因此八卡每个 epoch 约 10000 optimizer steps。若改设备数，需要同步调整 `data.train_batch_size` 或 `accumulate_grad_batches` 保持全局 batch。
