# Codex Hook 说明

`.codex/hooks.json` 把生命周期事件连接到 `agent-code/scripts/codex-hook.sh`。

hook 会把事件时间戳记录到 `docs/memory/hook-events.log`，把简短提醒追加到 `docs/memory/hook-reminders.log`，并在缺失时创建 `docs/memory/current.md` 或 `docs/memory/progress.md`。

hook 只负责提醒。Codex 仍必须判断哪些信息是持久信息，写入正确文档，并清理过期记忆。

共享文件系统上应保持高频 `PreToolUse` 和 `PostToolUse` hook 禁用。优先使用显式检查：

```sh
agent-code/scripts/agent/impact.sh
agent-code/scripts/agent/check.sh <scope>
```
