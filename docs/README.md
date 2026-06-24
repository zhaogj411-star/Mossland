# 文档索引

使用本索引恢复 Mossland 的持久上下文。

- `agent-start.md`：读取 `AGENTS.md` 后的第一个短上下文文件。
- `code-index.md`：代码、测试、命令和资源边界的紧凑地图。
- `agent/README.md`：agent harness 的用途和操作规则。
- `agent/commands.md`：用于验证的稳定命令入口。
- `mossland-codec.md`：读取 Mossland 多任务 codec、separation 预处理和数据集接法时使用。
- `music2latent-official-diff.md`：判断本地 `music2latent.yaml` 的 continuous/RVQ 训练路径与官方 `tmp/music2latent-training` 参考实现差异时使用。
- `music2latent-local-transformer-plan.md`：恢复用 `scripts/same` local transformer 替换 Music2Latent 卷积骨干、目标 `2*48000 -> 32*25` 且保留 skip connect 的设计方案时使用。
- `same-training.md`：判断 `scripts/same` 训练 wrapper 与 SAME 论文/Stable Audio Tools 公开训练代码的对齐程度、已实现 loss/discriminator 和剩余 auxiliary/stage 缺口时使用。
- `exp/same_flow_exp.md`：持续调试 `scripts/same_flow` 的 SAME-style Music2Latent 一致性重建实验记录、结构调整、评估命令和指标时使用。
- `exp/same_flow_music2latent_exp.md`：以 Music2Latent 官方训练过程为基准，逐步替换为 `scripts/same` 模块并比较 loss/梯度时使用。
- `music2latent.md`：读取独立 `scripts/music2latent` 训练包、YAML 参数、RVQ 旁路和 demo 输出约定时使用。
- `evaluation-metrics.md`：调研 `scripts/mossland_codec/tasks.py` 中 5 个任务的评估指标、论文依据和可复用代码仓库时使用。
- `eval-benchmark-report.md`：跟踪 Mossland eval pipeline、论文指标复现状态、真实运行结果和剩余缺口时使用。
- `eval-benchmark-adapters.md`：手动接入新评测模型、理解 `eval_benchmark/` 目录职责、启动 smoke/分片评测时使用。
- `superpowers/specs/2026-06-17-mossland-codec-time-aligned-rvq-design.md`：当前有效的 Mossland codec RVQ 设计，使用 local Transformer 输出 25Hz 时间对齐 token，连续表征为 32 维/token，离散表征为每 token 16 层 RVQ。
- `superpowers/plans/2026-06-17-mossland-codec-time-aligned-rvq-implementation.md`：当前有效的 time-aligned RVQ 实施计划，按 TDD 分阶段替换 FSQ、接入 local time-token encoder、动态 RVQ、训练/推理/demo 和文档更新。
- `superpowers/archive/specs/2026-06-17-mossland-codec-rvq-design.md`：已废弃的 512 维 chunk-summary RVQ 设计；仅用于了解为什么该方案不能解决 time alignment。
- `superpowers/archive/plans/2026-06-17-mossland-codec-rvq-implementation.md`：已废弃的 chunk-summary RVQ 实施计划；不要执行。见 `superpowers/archive/README.md`。
- `papers/README.md`：评估指标调研下载论文 PDF 的本地索引。
- `memory/current.md`：新会话的短活跃上下文。
- `memory/progress.md`：简洁的交接历史和下一步。
- `memory/inbox.md`：等待分拣的未归类 notes。
- `memory/maintenance.md`：持久记忆放置规则。
- `memory/codex-hooks.md`：hook 行为和约束。
