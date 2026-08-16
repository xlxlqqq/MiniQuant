import torch
import torch.nn as nn

from MiniQuant.quant.int4 import (
    GroupwiseInt4Tensor,
    quantize_groupwise_int4,
    dequantize_groupwise_int4,
    encode_int4,
    decode_int4,
    pack_int4,
    unpack_int4,
)

class WeightOnlyInt4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = True,
    ):
        super().__init__()

        if in_features % group_size != 0:
            raise ValueError("in features must be divisible by group size")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.num_groups = (in_features // group_size)

        self.register_buffer("packed_weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8))

        self.register_buffer("scale", torch.empty(out_features, self.num_groups, dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.bias = None
        
    def forward(self, x):
        W = self.dequantize_weight()

        return torch.nn.functional.linear(
            x,
            W,
            self.bias,
        )

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
    ):

        qlinear = cls(
            linear.in_features,
            linear.out_features,
            group_size,
            bias=linear.bias is not None,
        )

        W = linear.weight.detach()

        qtensor = quantize_groupwise_int4(
            W,
            group_size=group_size,
        )

        packed = pack_int4(
            qtensor.q
        )

        qlinear.packed_weight.copy_(
            packed.reshape(
                linear.out_features,
                linear.in_features // 2,
            )
        )

        qlinear.scale.copy_(
            qtensor.scale
        )

        if linear.bias is not None:
            qlinear.bias.copy_(
                linear.bias.detach()
            )

        return qlinear

    def dequantize_weight(self):

        q = unpack_int4(
            self.packed_weight.reshape(-1),
            self.out_features * self.in_features,
        )

        q = q.reshape(
            self.out_features,
            self.in_features,
        )

        q = q.reshape(
            self.out_features,
            self.num_groups,
            self.group_size,
        )

        scale = self.scale.unsqueeze(-1)

        W = q.float() * scale

        return W.reshape(
            self.out_features,
            self.in_features,
        )