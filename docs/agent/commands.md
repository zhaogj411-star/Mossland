# 代理命令

本文件是 Codex 验证用的稳定命令索引。

## 预检

```sh
agent-code/scripts/agent/preflight.sh
```

打印 Python、pytest、搜索工具、git 状态和推荐入口。

## 影响分析

```sh
agent-code/scripts/agent/impact.sh
```

报告变更文件，并建议对应的 `agent-code/scripts/agent/check.sh <scope>` 命令。

## 检查范围

```sh
agent-code/scripts/agent/check.sh agent-harness
agent-code/scripts/agent/check.sh scripts
agent-code/scripts/agent/check.sh docs
agent-code/scripts/agent/check.sh python
agent-code/scripts/agent/check.sh all
```

- `agent-harness`：验证代理测试和 hook/script shell 语法。
- `scripts`：验证 shell 脚本，并在存在代理测试时运行它们。
- `docs`：验证代理测试和 `git diff --check`。
- `python`：存在根目录 `tests/` 或 `agent-code/tests/` 时运行 pytest。
- `all`：按顺序运行以上 scope。

## Mossland codec 训练短跑

本机只有一张 RTX 4090 时，用以下命令验证 `scripts/train.py`、`mossland-codec.yaml`、训练 step、demo callback 和 checkpoint 保存链路。`scripts/train.py` 当前未启用 `rootutils.setup_root`，直接按文件路径执行时需要显式设置 `PYTHONPATH`。

```sh
PYTHONPATH=/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland \
HYDRA_FULL_ERROR=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false \
/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python \
  scripts/train.py \
  experiment=mossland-codec \
  trainer.devices='[0]' \
  trainer.strategy=auto \
  trainer.min_epochs=0 \
  trainer.max_epochs=1 \
  +trainer.max_steps=1 \
  trainer.num_sanity_val_steps=0 \
  checkpoint_every_n_train_steps=1 \
  data.num_workers=1 \
  data.pin_memory=false \
  callbacks.demo_callback.demo_num=1
```

注意：`data.num_workers=0` 会触发 PyTorch `prefetch_factor` 限制，因为当前 `Experiment_Dataset.train_dataloader()` 总是传 `prefetch_factor`；调试短跑至少使用 `data.num_workers=1`。
