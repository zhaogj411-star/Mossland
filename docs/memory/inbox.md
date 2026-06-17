# 收件箱

本文件用于暂存需要后续分拣到更好命名文档中的未归类记录。

- 2026-06-17 RVQ 最终方向：旧的 512-D chunk-summary RVQ 方案已废弃，因为 learned-query Transformer summary 没有实质时间对齐。当前有效目标是 `2 * 48000 Hz` stereo audio -> `25 Hz` latent token rate；continuous 表征为 `32` 维/token，即 `32 * 25Hz`；discrete 表征为每 token `16` 层 RVQ codebook（默认 `codebook_size=1024`，约 `4 kbps`），并支持动态 `n_quantizers <= 16`。encoder 需改为 local Transformer 或等价局部结构，输出固定时间位置锚定的 token；RVQ 在 token 维做。demo 每个样本只保存一个 concat wav：`src -> target -> rvq/discrete decode -> continuous decode`。当前有效设计见 `docs/superpowers/specs/2026-06-17-mossland-codec-time-aligned-rvq-design.md`。
