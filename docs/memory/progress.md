# 进展

本文件保存当前工作的简洁交接历史。

## 当前工作

- 已添加代理规则、hook、脚本、索引和测试，作为 Mossland 的初始 agent harness。
- 2026-06-08：仓库文档和 agent harness 面向人阅读的输出改为中文。
- 2026-06-08：`agent-code/scripts/agent/check.sh all` 通过，覆盖 agent harness、脚本语法、docs 检查和 pytest。
- 2026-06-08：将完全由机器管理的 harness 代码集中到 `agent-code/`；`.codex/hooks.json` 仍留在 `.codex/`，但命令指向 `agent-code/scripts/codex-hook.sh`。

## 下一步

- 未来变更后，运行 `agent-code/scripts/agent/impact.sh` 和它建议的检查。

## 阻塞项

- 无。

## Hook 活动

- 尚无需要保留的 hook 活动。

- 2026-06-08T08:18:19Z `UserPromptSubmit`: 如果任务状态变化，刷新当前工作、下一步和阻塞项。

- 2026-06-08T08:19:37Z `Stop`: 如果任务状态变化，刷新当前工作、下一步和阻塞项。

- 2026-06-08T08:20:31Z `UserPromptSubmit`: 如果任务状态变化，刷新当前工作、下一步和阻塞项。
