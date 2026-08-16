from dataclasses import dataclass

import torch


@dataclass
class GroupwiseInt4Tensor:
    q: torch.Tensor
    scale: torch.Tensor
    group_size: int

# int4 量化
def quantize_groupwise_int4(
    x: torch.Tensor,
    group_size: int = 128,
) -> GroupwiseInt4Tensor:

    if not x.is_floating_point():
        raise TypeError(
            "x must be floating point"
        )

    if x.ndim != 2:
        raise ValueError(
            "Only 2D tensors are supported"
        )

    out_features, in_features = x.shape

    if in_features % group_size != 0:
        raise ValueError(
            "in_features must be divisible "
            "by group_size"
        )

    num_groups = (
        in_features // group_size
    )

    x_grouped = x.reshape(
        out_features,
        num_groups,
        group_size,
    )

    # Symmetric INT4
    qmax = 7.0

    max_abs = x_grouped.abs().amax(
        dim=2,
        keepdim=True,
    )

    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / qmax,
    )

    q = torch.round(
        x_grouped / scale
    )

    q = torch.clamp(
        q,
        -7,
        7,
    )

    # Keep q as int8 for now.
    # PyTorch 普通 Tensor 并没有一个可以直接拿来存储任意权重的原生 torch.int4 dtype
    # 用 INT8 Tensor 暂时承载 INT4 的数值
    q = q.to(torch.int8)

    q = q.reshape_as(x)

    scale = scale.squeeze(-1)

    return GroupwiseInt4Tensor(
        q=q,
        scale=scale,
        group_size=group_size,
    )

# int4 反量化
def dequantize_groupwise_int4(
    qtensor: GroupwiseInt4Tensor,
) -> torch.Tensor:

    out_features, in_features = (
        qtensor.q.shape
    )

    group_size = qtensor.group_size

    num_groups = (
        in_features // group_size
    )

    q = qtensor.q.reshape(
        out_features,
        num_groups,
        group_size,
    )

    scale = qtensor.scale.unsqueeze(-1)

    x = q.float() * scale

    return x.reshape(
        out_features,
        in_features,
    )