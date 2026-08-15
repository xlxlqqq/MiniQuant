import torch
from dataclasses import dataclass

@dataclass
class PerChannelQuantizedTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    channel_dim: int

def symmetric_quantize_per_channel_int8(
    x: torch.Tensor,
    channel_dim: int = 0,
) -> PerChannelQuantizedTensor:
    if x.dim() == 0:
        raise ValueError("x must have at least one dimension.")
    
    channel_dim = channel_dim % x.dim()

    qmax = 127.0

    # 规约，x == (out_channels, in_channels)的时候，reduce_dims == (1,)
    reduce_dims = tuple(
        dim for dim in range(x.ndim) if dim != channel_dim
    )

    # 计算每个通道的最大绝对值
    # keepdim=True 保留维度，方便后续广播[4,8]变成[4,1]，而不是[4]，否则会报错
    max_abs = x.abs().amax(dim = reduce_dims, keepdim=True)

    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)

    q = torch.round(x / scale)

    q = torch.clamp(q, -qmax, qmax).to(torch.int8)

    zero_point = torch.zeros_like(scale, dtype=torch.int32)

    return PerChannelQuantizedTensor(q=q, scale=scale, zero_point=zero_point, channel_dim=channel_dim)

def symmetric_dequantize_per_channel_int8(
    qtensor: PerChannelQuantizedTensor,
) -> torch.Tensor:
    return qtensor.q.float() * qtensor.scale