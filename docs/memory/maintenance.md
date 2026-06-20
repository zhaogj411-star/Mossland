# 记忆维护

使用 `docs/` 保存持久项目上下文。

- 短活跃状态放入 `docs/memory/current.md`。
- 简洁交接历史放入 `docs/memory/progress.md`。
- 未归类 notes 放入 `docs/memory/inbox.md`；目标位置清楚后，把稳定知识移入命名更好的文档。
- 新增、移动、重命名或废弃持久文档时，更新 `docs/README.md`。
- 代码布局、入口点、配置、测试或资源边界变化时，更新 `docs/code-index.md`。
- 删除、合并或标记过期文档，避免未来会话恢复过时上下文。
- 永远不要存储 secrets、凭据、token 或个人机器状态。

## 归档约定

- 不再参与当前训练/入口、但需保留历史参考的代码移入 `scripts/_archive/`；对应实验配置移入 `scripts/configs/experiment/_archive/`；针对已废弃 API 的测试移入 `tests/_archive/`。
- 已废弃的设计 spec/plan 移入 `docs/superpowers/archive/`。
- `pytest.ini` 的 `norecursedirs` 已排除所有 `*_archive`，归档代码不参与测试采集。
- 归档时在 `scripts/_archive/` 或 archive README 写明被取代原因与替代入口，并同步更新 `docs/code-index.md`、`docs/README.md`。

## 共享模块约定

- codec 实验共享的 `quantize.py` / `training_base.py` 单一来源在 `scripts/codec_common/`，新增 codec 实验从这里 `import`，不要再逐字节复制到各模型目录。
