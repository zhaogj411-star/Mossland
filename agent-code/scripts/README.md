# 脚本

本目录保存命令行封装脚本和批处理脚本。

- `codex-hook.sh` 和 `codex-post-tool-memory.sh`：Codex 生命周期 hook。
- `agent/preflight.sh`：打印当前代理环境和 git 摘要。
- `agent/impact.sh`：根据当前变更推荐验证 scope。
- `agent/check.sh`：按 scope 运行固定验证命令。

常用代理命令：

```sh
agent-code/scripts/agent/preflight.sh
agent-code/scripts/agent/impact.sh
agent-code/scripts/agent/check.sh --help
```

agent harness 规则见 `docs/agent/README.md`。
