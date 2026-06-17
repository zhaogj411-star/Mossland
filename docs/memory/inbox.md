# 收件箱

本文件用于暂存需要后续分拣到更好命名文档中的未归类记录。

- 2026-06-17：用户希望把 `scripts/mossland-codec` 的 FSQ 换成 RVQ。核心意图是把一个 `spec_length` chunk 当前的 128 个 4 维 bottleneck latent 展平成一个 512 维时间步，在这 512 维上做 RVQ。RVQ 默认参数定为 `codebook_size=1024`、`num_quantizers=256`：每 chunk 约 `2560 bits`，比当前 FSQ 的 `128 * 4 * log2(11) = 1771 bits/chunk` 高约 1.45 倍；这是用户确认的取舍。连续表征训练应尽量保留；推荐方案是先旁路训练 RVQ 还原连续 512 维，再让 RVQ 像现有 FSQ 一样以一定概率进入训练路径并参与 decoder conditioning。参考实现可看 Descript DAC 的量化模块。
