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

# 范围从 [-7, 7] 映射到 [1, 15]
def encode_int4(
    q: torch.Tensor,
) -> torch.Tensor:

    if q.dtype != torch.int8:
        raise TypeError(
            "q must be int8"
        )

    if q.min() < -7 or q.max() > 7:
        raise ValueError(
            "q must be in [-7, 7]"
        )

    return (q + 8).to(torch.uint8)

# 范围从 [1, 15] 映射到 [-7, 7]
def decode_int4(
    encoded: torch.Tensor,
) -> torch.Tensor:

    if encoded.dtype != torch.uint8:
        raise TypeError(
            "encoded must be uint8"
        )

    if encoded.min() < 1 or encoded.max() > 15:
        raise ValueError(
            "encoded must be in [1, 15]"
        )

    return encoded.to(torch.int16) - 8

# 两个 int4 占用一个byte
# 
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    if q.dtype != torch.int8:
        raise TypeError("q must be int8")
    
    if q.numel() % 2 != 0:
        raise ValueError("Number of Int 4 values must be even.")

    encoded = encode_int4(q)

    flat = encoded.reshape(-1)

    # 隔两个取数，从0开始
    low = flat[0::2]
    # 隔两个取数，从1开始
    high = flat[1::2]

    packed = (low | (high << 4))

    return packed

# unpack
def unpack_int4(
    packed: torch.Tensor,
    num_values: int,
) -> torch.Tensor:

    if packed.dtype != torch.uint8:
        raise TypeError(
            "packed must be uint8"
        )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    encoded = torch.empty(
        packed.numel() * 2,
        dtype=torch.uint8,
        device=packed.device,
    )

    encoded[0::2] = low
    encoded[1::2] = high

    encoded = encoded[:num_values]

    return decode_int4(
        encoded
    ).to(torch.int8)

# 使用int4 packed 和 group wise进行量化
def quantize_groupwise_int4_packed(
    x: torch.Tensor,
    group_size: int = 128,
):
    qtensor = quantize_groupwise_int4(
        x,
        group_size=group_size,
    )

    packed = pack_int4(
        qtensor.q
    )

    return packed, qtensor.scale