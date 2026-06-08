# 代理 Harness

本目录索引可复用的代理命令和规则。

- 启动时先读 `AGENTS.md`，再读 `docs/agent-start.md`。
- 使用 `agent-code/scripts/agent/preflight.sh` 恢复本地环境状态。
- 选择验证命令前，先使用 `agent-code/scripts/agent/impact.sh`。
- 使用 `agent-code/scripts/agent/check.sh <scope>`，不要从零猜测检查命令。
- 持久上下文保存在 `docs/` 下，不依赖私有聊天历史。
- 面向人阅读的 harness 说明、hook 提醒、脚本输出和未来 harness 设定默认使用中文；命令名、路径、scope 和代码标识保留英文接口。

除非内容描述的是当前 Mossland 规则、决策或工作流，不要把来源项目的长篇文档复制到 Mossland。
