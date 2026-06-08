# Agent Code

本目录集中存放完全由 Codex/agent harness 管理的机器代码。

- `scripts/`：hook、预检、影响分析和检查脚本。
- `tests/`：agent harness 自检测试。

`.codex/hooks.json` 必须留在 `.codex/`，但其中命令应指向本目录下的脚本。项目记忆仍保存在根目录 `docs/`。
