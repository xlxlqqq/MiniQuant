from dataclasses import dataclass

import torch

@dataclass
class QuantizedTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor

# 量化
def symmetric_quantize_int8(
    x: torch.Tensor,
) -> QuantizedTensor:
    '''
    Args:
        x: A floating-point tensor to be quantized.
    Returns:
        QuantizedTensor:
            q: INT8 tensor
            scale: scale factor
            zero_point: zero point (always 0 for symmetric quantization)
    '''
    if not x.is_floating_point():
        raise TypeError("Input tensor must be a floating-point tensor.")
    
    qmax = 127.0

    max_abs = x.abs().max()

    # 防止分母为 0 
    scale = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs / qmax,
    )

    q = torch.round(x / scale)

    q = torch.clamp(q, -qmax, qmax)

    q = q.to(torch.int8)

    zero_point = torch.zeros(
        (),
        dtype=torch.int32,
        device = x.device,
    )

    return QuantizedTensor(q=q, scale=scale, zero_point=zero_point)

# 反量化
def symmetric_dequantize_int8(
    qtensor: QuantizedTensor,
) -> torch.Tensor:
    """
    Dequantize INT8 tensor back to floating point.
    """

    return qtensor.q.float() * qtensor.scale