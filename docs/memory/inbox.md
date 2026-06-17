# 收件箱

本文件用于暂存需要后续分拣到更好命名文档中的未归类记录。

- 2026-06-17：用户希望把 `scripts/mossland-codec` 的 FSQ 换成 RVQ。核心意图是把一个 `spec_length` chunk 当前的 128 个 4 维 bottleneck latent 展平成一个 512 维时间步，在这 512 维上做 RVQ。RVQ 默认参数定为 `codebook_size=1024`、最大 `num_quantizers=256`：用满 256 层时每 chunk 约 `2560 bits`，比当前 FSQ 的 `128 * 4 * log2(11) = 1771 bits/chunk` 高约 1.45 倍；同时需要像 DAC 一样支持动态 `n_quantizers <= 256`。连续表征训练应尽量保留；推荐方案是先旁路训练 RVQ 还原连续 512 维，再让 RVQ 像现有 FSQ 一样以一定概率进入训练路径并参与 decoder conditioning。demo 每个样本只保存一个 concat wav，顺序为 `src -> target -> rvq/discrete decode -> continuous decode`。参考实现可看 Descript DAC 的量化模块。
- 2026-06-17 开放设计点：用户担心“一个 chunk 一个 512-D summary embedding”没有实质性时间顺序，长音频 decode 时必须按 encode 的 chunk 边界切回去，不友好。候选改法是把一个 chunk 内的 128 个 4-D latent 改成 time-aligned layout，例如按当前 frontend `time_dim=8` 拆成 8 个时间步，每步 16 个 4-D latent，即每步 64-D；RVQ 按时间步做（例如每步最多 32 个 1024-size codebook，合计仍是旧 chunk 的 256 个 codebook），decode 内部再按 8 个 latent time steps 组窗。
- 2026-06-17 设计修正：用户确认 RVQ 不必在整 chunk 的 512 维 summary 上做，可以在 token 维上做。后续 RVQ 方案应优先考虑 time-aligned token RVQ：保留或生成具有时间顺序的 latent token，再对每个 token 的向量做 RVQ，而不是先把整 chunk 聚合成单个无内部顺序的 512-D embedding。
