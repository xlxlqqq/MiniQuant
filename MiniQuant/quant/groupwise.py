from dataclasses import dataclass

import torch

@dataclass
class GroupwiseQuantizedTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    group_size: int
    # channel_dim 用来标识：scale / zero_point 是沿着 tensor 的哪个维度“按 channel 分的
    channel_dim: int

# 量化
def symmetric_quantize_groupwise_int8(
    x: torch.Tensor,
    group_size: int,
    channel_dim: int = 0,
) -> GroupwiseQuantizedTensor:
    if not x.is_floating_point():
        raise TypeError("X must be a floating point tensor")
    
    if x.ndim != 2:
        raise ValueError("Currently only 2D tensors are supported!")

    channel_dim = channel_dim % x.ndim

    if channel_dim != 0:
        raise ValueError("For linear weights, channel dim must be 0.")

    out_features, in_features = x.shape

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be "
            f"divisible by group_size ({group_size})"
        )
    
    num_groups = in_features // group_size

    '''
        W.shape = [4096, 4096]
        group_size = 128
        则 x_grouped [4096, 32, 128]
    '''
    x_grouped = x.reshape(
        out_features,
        num_groups,
        group_size,
    )
    qmax = 127.0

    max_abs = x_grouped.abs().amax(
        dim = 2,
        keepdim = True,
    )

    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / qmax,
    )

    q_grouped = torch.round(x_grouped / scale)

    q_grouped = torch.clamp(q_grouped, -127, 127).to(torch.int8)

    q = q_grouped.reshape_as(x)

    scale = scale.squeeze(-1)

    zero_point = torch.zeros_like(scale, dtype=torch.int32)

    return GroupwiseQuantizedTensor(
        q = q,
        scale = scale,
        zero_point = zero_point,
        group_size = group_size,
        channel_dim = channel_dim,
    )

# 反量化
def symmetric_dequantize_groupwise_int8(
    qtensor: GroupwiseQuantizedTensor,
) -> torch.Tensor:

    out_features, in_features = qtensor.q.shape
    group_size = qtensor.group_size

    num_groups = in_features // group_size

    # reshape 成 (OC, groups, group_size)
    q_grouped = qtensor.q.reshape(
        out_features,
        num_groups,
        group_size,
    )

    # scale: [OC, groups] -> [OC, groups, 1]
    scale = qtensor.scale.unsqueeze(-1).float()

    # symmetric: 没有 zero_point
    x_grouped = q_grouped.float() * scale

    # 恢复原始 shape
    return x_grouped.reshape(
        out_features,
        in_features,
    )