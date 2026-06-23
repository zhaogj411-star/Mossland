"""连续时间一致性模型（sCM / sCT）的训练数学库。

实现了 Lu & Song 论文《Simplifying, Stabilizing and Scaling Continuous-Time
Consistency Models》(arXiv:2410.11081, ICLR 2025 Oral) 中的 **TrigFlow（三角流）
参数化** 与 **连续时间一致性训练目标**。

背景与变体选择
--------------
经典一致性模型（CM）用 *离散* 的相邻噪声点做有限差分来近似"沿 PF-ODE 轨迹
输出保持一致"这一约束；本论文改为在连续时间上直接用 **JVP（前向模式自动微分）**
求出精确的时间切向量 ``df/dt``，从而消除有限差分带来的偏差与不稳定。

由于本 codec 是 **从零训练**、没有预训练的扩散教师模型，因此采用 **sCT** 变体：
概率流常微分方程（PF-ODE）的速度场用无偏估计
``dx_t/dt = cos(t) z - sin(t) x0`` 直接给出，而不是去查询教师网络
（那是 sCD 蒸馏变体的做法）。

坐标系约定
----------
唯一的"时间/噪声"变量是 TrigFlow 时间 ``t ∈ (t_min, t_max] ⊂ (0, π/2]``：
``t→0`` 对应干净数据，``t→π/2`` 对应纯噪声先验 ``N(0, sigma_data^2)``。
区间端点 ``t_min`` / ``t_max`` 直接在配置里以时间域给出（不再从 EDM 的
``sigma_min`` / ``sigma_max`` 转换而来）。``sigma_data`` 仍保留，因为它是 TrigFlow
自身的参数（出现在 ``c_in = 1/sigma_data``、一致性函数 ``f_theta`` 与先验尺度里）。

工程约定
--------
本模块刻意不依赖 ``models.py``，以保持导入图无环：``models`` 与 ``wrapper``
都从这里 import，而这里不反向 import 它们。
"""

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


# --------------------------------------------------------------------------- #
# 时间广播工具
# --------------------------------------------------------------------------- #
def _broadcast_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """把逐样本的 ``[B]`` 时间张量 reshape 成 ``[B, 1, ..., 1]``，便于与
    ``[B, C, F, T]`` 这类特征张量做逐元素广播。

    Args:
        t: 形状 ``[B]`` 的张量。
        ndim: 目标特征张量的维度数（例如频谱图为 4）。
    """
    return t.reshape(t.shape[0], *([1] * (ndim - 1)))


# --------------------------------------------------------------------------- #
# 时间采样 与 前向加噪过程
# --------------------------------------------------------------------------- #
def sample_t(
    batch_size: int,
    *,
    sigma_data: float,
    p_mean: float,
    p_std: float,
    t_min: float,
    t_max: float,
    device: torch.device,
) -> torch.Tensor:
    """logit-normal（对数正态）时间采样。

    流程对应论文 Algorithm 1 的第一步：先采 ``tau ~ N(p_mean, p_std)``
    （此时 ``e^tau`` 恰好扮演噪声尺度 ``sigma`` 的角色），再经 TrigFlow 双射
    ``t = arctan(e^tau / sigma_data)`` 转换到时间域。等价于对 ``tan(t)`` 施加
    对数正态先验（即 Karras et al. 2022 的噪声采样分布）。

    采样结果会被 clamp 到 ``[t_min, t_max]``（由配置直接给定的时间区间端点）。

    Args:
        batch_size: 批大小。
        sigma_data: 数据标准差（TrigFlow 中先验噪声的尺度）。
        p_mean: 高斯提议分布在 log 域的均值（控制采样集中在哪个噪声水平）。
        p_std: 高斯提议分布在 log 域的标准差（控制噪声水平的覆盖范围）。
        t_min: 时间下界（接近 0，对应近干净数据）。
        t_max: 时间上界（接近 π/2，对应近纯噪声）。
        device: 输出张量所在设备。

    Returns:
        形状 ``[batch_size]`` 的时间张量。
    """
    tau = torch.randn(batch_size, device=device) * p_std + p_mean
    t = torch.arctan(torch.exp(tau) / sigma_data)
    return t.clamp(t_min, t_max)


def trigflow_noise(x0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """TrigFlow 前向加噪过程 ``x_t = cos(t) x0 + sin(t) z``。

    这是论文的核心几何：干净数据 ``x0`` 与噪声 ``z`` 被放在单位圆上做插值，
    ``t=0`` 时纯数据，``t=π/2`` 时纯噪声。注意噪声 ``z`` 的尺度应为 ``sigma_data``
    （即 ``z ~ N(0, sigma_data^2 I)``），调用方负责按此采样。

    Args:
        x0: 干净数据，形状 ``[B, C, F, T]``。
        z: 同形状噪声。
        t: 逐样本时间 ``[B]``，函数内部会广播到 ``x0`` 的形状。
    """
    t = _broadcast_t(t, x0.ndim)
    return torch.cos(t) * x0 + torch.sin(t) * z


# --------------------------------------------------------------------------- #
# 连续时间一致性切向量（sCM 的核心）
# --------------------------------------------------------------------------- #
def consistency_tangent(
    network_fn,
    x_t: torch.Tensor,
    t: torch.Tensor,
    *,
    dxt_dt: torch.Tensor,
    sigma_data: float,
    warmup_r: float,
    norm_const: float,
):
    """计算（已 detach 的）网络输出与 sCM 训练切向量 ``g``。

    切向量遵循 arXiv:2410.11081 Algorithm 1 中经过重排（rearranged）的形式::

        g = -cos^2(t) * (sigma_data * F- - dx_t/dt)
            - r * cos(t) * sin(t) * (x_t + sigma_data * dF-/dt)

    含义解析：
      * ``F-`` 是停止梯度参数 ``theta^-`` 处的网络输出（即论文中的 EMA/同参教师）；
      * ``dF-/dt`` 是网络对 ``(x_t, t)`` 的 **精确全时间导数**，由 JVP 沿切向
        ``(dx_t/dt, 1)`` 一次前向传播算得 —— 这正是连续时间一致性相对离散有限差分
        的关键改进；
      * ``r`` 是切向量预热（tangent warmup）系数：训练初期把不稳定的第二项
        （含 ``sin(t)`` 与二阶导信息）按 ``r`` 线性放大 0→1，论文用前 1 万步预热，
        以稳定早期训练；
      * 重排后第一项的 ``cos^2(t)`` 与第二项的 ``cos(t)sin(t)`` 系数，已经把原始公式里
        ``cos(t) * df_θ-/dt`` 这一项吸收进来（同时配合 JVP 在 fp16 下的数值稳定写法）。

    两个返回张量都被 detach：``g`` 作为回归目标，JVP 也是在 ``theta^-`` 处求值，
    因此本函数不向模型参数回传梯度（梯度只通过 wrapper 里另一支可训练前向传播流入）。

    数值实现要点：JVP 在 fp32、且强制使用 **MATH 版 SDPA 注意力后端** 下运行 ——
    因为 flash / memory-efficient 两种融合注意力 kernel 没有前向模式自动微分（forward-AD）
    规则，直接跑会报错。

    Args:
        network_fn: 纯函数 ``(x_t, t) -> F_theta(x_t / sigma_data, t)``，
            通过闭包捕获固定的条件信息（latents / features）。
        x_t: 加噪输入 ``[B, C, F, T]``。
        t: 逐样本时间 ``[B]``。
        dxt_dt: PF-ODE 速度场估计 ``cos(t) z - sin(t) x0``（sCT 变体），与 ``x_t`` 同形状。
        sigma_data: 数据标准差。
        warmup_r: 切向量预热系数 ``r ∈ [0, 1]``。
        norm_const: 切向量归一化里的常数 ``c``，见下方 ``g / (||g|| + c)``。

    Returns:
        ``(F_detached, g)``：``theta^-`` 处的网络输出，以及归一化后的切向量；
        二者均为 fp32 且已 detach。
    """
    # 整个切向量计算转 fp32，避免 bf16/fp16 下 JVP 的数值问题。
    x_t = x_t.float()
    dxt_dt = dxt_dt.float()
    t = t.float()

    # 关闭 autocast 并强制 MATH 注意力后端：JVP（前向模式 AD）只在 MATH kernel 上有定义。
    with torch.autocast(device_type=x_t.device.type, enabled=False):
        with sdpa_kernel(SDPBackend.MATH):
            # 一次 JVP 同时拿到：前向输出 F-（f_out）与沿 (dx_t/dt, 1) 的方向导数 dF-/dt（df_dt）。
            # 时间分量的切向取 1（ones_like(t)），对应 d t / d t = 1。
            f_out, df_dt = torch.func.jvp(
                network_fn,
                (x_t, t),
                (dxt_dt, torch.ones_like(t)),
            )

    # 教师分支整体停止梯度。
    f_out = f_out.detach()
    df_dt = df_dt.detach()

    # 把时间广播成特征形状，预先算好 cos/sin。
    t_b = _broadcast_t(t, x_t.ndim)
    cos_t, sin_t = torch.cos(t_b), torch.sin(t_b)

    # 重排后的切向量（见上方 docstring 公式）。
    g = -(cos_t**2) * (sigma_data * f_out - dxt_dt) - warmup_r * cos_t * sin_t * (
        x_t + sigma_data * df_dt
    )

    # 逐样本在特征维上做归一化 g / (||g|| + c)，抑制不同噪声水平下切向量量级差异过大，
    # 提升尺度稳定性（论文中的 tangent normalization；norm_const 即常数 c，默认 0.1）。
    norm = g.flatten(1).norm(dim=1)
    norm = _broadcast_t(norm, x_t.ndim)
    g = g / (norm + norm_const)
    return f_out, g


# --------------------------------------------------------------------------- #
# 自适应（不确定性）权重 w_phi(t)
# --------------------------------------------------------------------------- #
class _TimeEmbedding(nn.Module):
    """对标量时间做位置编码（positional embedding），风格与 codec 主干一致。

    采用低频率尺度的正余弦编码（而非高频 Fourier 特征）：论文指出，时间嵌入对
    ``t`` 的导数会随 Fourier 尺度增大而放大，从而在连续时间训练中引入不稳定，
    因此这里用接近位置编码的低尺度形式。
    """

    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 构造一组从 1 衰减到 1/max_positions 的频率（log 均匀分布）。
        freqs = torch.arange(self.dim // 2, device=t.device, dtype=torch.float32)
        freqs = freqs / max(1, self.dim // 2 - 1)
        freqs = (1.0 / self.max_positions) ** freqs
        # 外积后拼接 sin / cos，得到 [B, dim] 的时间嵌入。
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaptiveWeight(nn.Module):
    """EDM2 风格的可学习不确定性权重 ``w_phi(t)``。

    与主模型联合训练。在损失中，``e^{w_phi(t)}`` 作为乘性权重、``- w_phi(t)`` 作为
    加性正则项（见 :func:`scm_loss`）。二者配合等价于让网络自动学到"每个噪声水平 t
    上损失的不确定度"，从而 **自动平衡不同 t 的损失方差**，免去手工调权重调度的麻烦。

    直觉：若某个 t 上损失天然偏大（难学），模型会学到较大的 ``w_phi``，从而压低该项
    的乘性权重 ``e^{-w_phi}``（注意优化的是 ``e^{w}·L - w``，对 w 求极值后有效权重 ∝ 1/L），
    使各噪声水平对总梯度的贡献更均衡。
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.embed = _TimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # 输入逐样本时间 [B]，输出逐样本标量权重 [B]。
        return self.mlp(self.embed(t)).squeeze(-1)


# --------------------------------------------------------------------------- #
# sCM 损失
# --------------------------------------------------------------------------- #
def scm_loss(
    f_pred: torch.Tensor,
    f_detached: torch.Tensor,
    g: torch.Tensor,
    w_phi: torch.Tensor,
):
    """带可学习不确定性权重的连续时间一致性损失。

        L = mean_t[ e^{w_phi(t)} * || f_theta - f- + g ||^2 - w_phi(t) ]

    其中：
      * ``f_pred`` 是可训练的一致性函数输出（clean estimate），携带梯度；
      * ``f_detached`` 与 ``g`` 都已 detach，共同构成回归目标 ``f_detached - g``；
      * ``g`` 已按逐样本 L2 norm 归一化，因此优化时使用平方范数 ``sum``；
        per-element MSE 仅作为与张量尺寸无关的监控指标。

    可以理解为：让可训练网络去拟合"教师输出沿轨迹移动一个无穷小步后的值"，
    而该位移正是切向量 ``g`` 所刻画的方向。

    Args:
        f_pred: 可训练一致性函数输出 ``[B, C, F, T]``。
        f_detached: 教师（theta^-）一致性函数输出，已 detach。
        g: :func:`consistency_tangent` 给出的归一化切向量，已 detach。
        w_phi: :class:`AdaptiveWeight` 输出的逐样本权重 ``[B]``。

    Returns:
        ``(loss, metrics)``：标量损失，以及用于日志记录的指标字典。
    """
    # ``g`` follows the increasing-noise direction. The high-noise student should
    # match the teacher moved toward the lower-noise endpoint, i.e. f^- - g.
    residual = f_pred.float() - f_detached.float() + g.float()
    residual_sq = residual.pow(2).flatten(1)
    # ``g`` is L2-normalized per sample, so the natural unweighted target scale is
    # O(1). Using mean over all feature dimensions would shrink gradients by the
    # representation size and prevent the codec from starting to use its latents.
    per_sample_sse = residual_sq.sum(dim=1)
    per_sample_mse = residual_sq.mean(dim=1)
    # 不确定性加权：乘性 e^{w} 抑制/放大，加性 -w 防止 w 退化到无穷大。
    weighted = torch.exp(w_phi) * per_sample_sse - w_phi
    loss = weighted.mean()
    metrics = {
        "loss/consistency_mse": per_sample_mse.mean().detach(),  # 未加权的原始 MSE，便于观察收敛
        "loss/consistency_sse": per_sample_sse.mean().detach(),  # 实际优化的未加权平方范数
        "loss/adaptive_weight": w_phi.mean().detach(),           # 学到的平均权重
        "tangent/norm_mean": g.flatten(1).norm(dim=1).mean().detach(),  # 切向量平均范数（监控数值稳定性）
    }
    return loss, metrics
