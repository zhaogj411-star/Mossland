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

## CodeGraph

本机已安装 CodeGraph CLI `1.0.1`，路径为 `/root/.local/bin/codegraph`，standalone bundle 在 `/root/.codegraph/versions/v1.0.1`。Codex 全局 MCP 配置已由 `codegraph install --target codex --location global --yes` 写入；新 Codex 会话重启后可用 MCP 工具，当前或非 MCP 场景可直接用 shell 命令。

代码导航、调用关系和 blast radius 优先使用：

```sh
codegraph status .
codegraph query MosslandCodecTransformer
codegraph explore "MosslandCodecTransformer generate_waveform"
codegraph node scripts/mossland-codec/models.py
codegraph callers generate_waveform
codegraph impact MosslandCodecTransformer
```

注意：`codegraph init .` 已在本仓库创建 `.codegraph/`，该目录被 `.gitignore` 忽略。CodeGraph 1.0 会把 `.gitignore` 隐藏的嵌入式 git repo 也纳入索引，当前 `tmp/` 下 baseline/metric refs 可能出现在结果和 `.codegraph/errors.log` 中；判断 Mossland 主项目逻辑时优先看 `scripts/`、`tests/`、`agent-code/` 和 `docs/` 路径。

## Claude Code

本机已全局安装 Claude Code，路径为 `/usr/local/bin/claude`。2026-06-19 验证版本为 `2.1.183 (Claude Code)`。

当前环境默认注入 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 及小写同名变量；安装或升级 Claude Code 时如需绕过代理，使用：

```sh
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  npm install -g @anthropic-ai/claude-code@latest --proxy=false --https-proxy=false
```

验证：

```sh
which claude
claude --version
npm list -g @anthropic-ai/claude-code --depth=0
```

AI Gateway 接入配置在 `~/.claude/settings.json`，使用 `ANTHROPIC_BASE_URL=https://aigw.sotatts.online` 和 `ANTHROPIC_AUTH_TOKEN`，不要把完整 token 写入仓库。2026-06-19 排查结论：本机配置已生效，`/v1/models` 会列出 `claude-opus-4-7`、`claude-opus-4-8` 等 Claude 模型，但同一 token 直接请求 `/v1/messages` 会返回 `no available upstream channel for model ...`，所以 Claude Code 报 “selected model may not exist or you may not have access”。`gpt-5.4`、`gpt-5.4-mini`、`gpt-5.5` 和 `codex-auto-review` 在网关的 Anthropic messages 入口可直连，但 Claude Code 发送的 system role 会被这些后端拒绝，不能作为 Claude Code 的直接替代。需要网关侧为 `claude-*` 模型启用 upstream，或修复非 Claude 模型的 Anthropic 兼容转换。

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

## Mossland codec H200 启动

训练平台的启动命令必须指向脚本或 Python 命令，不能填 `logs/.../runs/<timestamp>` 输出目录。若把 run 目录填成命令，会得到 `<run_dir>: Is a directory`，这是 shell 启动层错误，不是训练代码报错。

单节点 H200 默认入口：

```sh
cd /inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/Mossland
bash/start_train.sh
```

等价展开：

```sh
PYTHONPATH=$PWD WANDB_MODE=offline HYDRA_FULL_ERROR=1 \
/inspire/qb-ilm2/project/embodied-multimodality/public/zhaoguojie/py_env/bin/python \
  -m scripts.train experiment=mossland-codec trainer.devices=8
```

恢复训练时 `ckpt_path` 或 `resume_from_ckpt` 必须指向 checkpoint 文件或 Lightning checkpoint 目录，不是 run 输出目录。`logs/mossland-codec/runs/2026-06-17_14-07-56` 只跑到几百 step，未到 `checkpoint_every_n_train_steps=5000`，因此没有可恢复的 `checkpoints/`。
