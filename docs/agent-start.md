# 代理启动胶囊

本文件是 Codex 会话的短启动上下文。

## 启动步骤

1. 读取 `AGENTS.md`。
2. 读取本文件。
3. 读取 `docs/memory/current.md`。
4. 代码任务读取 `docs/code-index.md`。
5. 当环境或依赖状态不清楚时，运行 `agent-code/scripts/agent/preflight.sh`。

## 当前形态

- Mossland 当前是轻量仓库，包含代理基础设施、文档记忆，还没有已提交的应用实现。
- 在 Mossland 专属代码、资源和工作流出现前，保持复制来的基础设施通用。

## 常用入口

```sh
agent-code/scripts/agent/preflight.sh
agent-code/scripts/agent/impact.sh
agent-code/scripts/agent/check.sh agent-harness
agent-code/scripts/agent/check.sh docs
```

## 收尾规则

结束实质性工作前，把当前状态、下一步和阻塞项更新到 `docs/memory/current.md`。较长历史移到 `docs/memory/progress.md`。
