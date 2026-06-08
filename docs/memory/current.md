# 当前上下文

本文件是新 Codex 会话的短活跃上下文。保持简洁。

## 当前工作

- Mossland 代理基础设施已从 `moss-training-framework` 的通用规则迁移而来。
- 仓库文档和 agent harness 面向人阅读的输出已统一改为中文。

## 稳定决策

- 使用 `docs/` 作为持久项目记忆根目录。
- 在 Mossland 专属代码出现前，保持导入的 agent harness 通用。
- 除非内容已成为当前 Mossland 需求、决策或工作流，不复制训练框架的长篇文档。
- 面向人阅读的文档、agent harness 说明、hook 提醒、脚本输出和未来 harness 设定默认使用中文；命令名、路径、scope、事件名、JSON key 和代码标识保留英文接口。
- 完全由机器管理的 harness 代码集中在 `agent-code/`；`.codex/hooks.json` 仍留在 `.codex/` 作为 Codex 配置入口。

## 下一步

- 开始添加 Mossland 专属代码时，更新 `docs/code-index.md`。
- 未来新增或修改文档、hook 提醒、脚本输出和 harness 设定时，默认继续使用中文。
- 验证入口使用 `agent-code/scripts/agent/check.sh <scope>`。

## 阻塞项

- 无。
