# 代码索引

Mossland 目前还没有已提交的应用实现代码。

## 仓库根部

- `AGENTS.md`：仓库代理规则。
- `.codex/hooks.json`：Codex 生命周期 hook 配置；命令指向 `agent-code/scripts/codex-hook.sh`。
- `agent-code/`：完全由机器管理的 agent harness 代码根目录。
- `agent-code/scripts/agent/`：显式 preflight、impact 和验证命令。
- `agent-code/scripts/codex-hook.sh`：生命周期提醒 hook。
- `agent-code/tests/agent/`：agent harness 测试。
- `docs/`：持久项目记忆。

新增、移动或废弃代码、入口点、配置、测试或资源边界时，更新本文件。
