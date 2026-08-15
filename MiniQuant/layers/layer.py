import torch
import torch.nn as nn

from miniquant.quant.affine import (
    symmetric_quantize_int8, 
    symmetric_dequantize_int8,
    QuantizedTensor,
)

class QuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weight 是一个可训练的参数，形状为 (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            # bias 为 None 时，注册一个名为 "bias" 的参数，但值为 None
            self.register_parameter(
                "bias",
                None,
            )
        
        # 初始化权重和偏置参数
        self.reset_parameters()

        # 量化后的权重参数，初始为 None
        self.weight_quantized: QuantizedTensor | None = None
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.weight,
            a = 5 ** 0.5,
        )

        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def quantize_weight(self):
        # 量化权重参数，并将结果存储在 self.weight_quantized 中
        # detach() 方法用于创建一个新的张量，该张量与原始张量共享相同的数据，但不会计算梯度。
        self.weight_q = symmetric_quantize_int8(self.weight.detach())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_q is None:
            raise RuntimeError(
                "Weight has not been quantized. Call quantize_weight() before forward()."
            )
        
        weight = symmetric_dequantize_int8(self.weight_q)

        return torch.nn.functional.linear(
            x,
            weight,
            self.bias,
        )