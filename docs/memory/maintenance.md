# 记忆维护

使用 `docs/` 保存持久项目上下文。

- 短活跃状态放入 `docs/memory/current.md`。
- 简洁交接历史放入 `docs/memory/progress.md`。
- 未归类 notes 放入 `docs/memory/inbox.md`；目标位置清楚后，把稳定知识移入命名更好的文档。
- 新增、移动、重命名或废弃持久文档时，更新 `docs/README.md`。
- 代码布局、入口点、配置、测试或资源边界变化时，更新 `docs/code-index.md`。
- 删除、合并或标记过期文档，避免未来会话恢复过时上下文。
- 永远不要存储 secrets、凭据、token 或个人机器状态。
