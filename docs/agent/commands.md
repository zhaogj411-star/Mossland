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
