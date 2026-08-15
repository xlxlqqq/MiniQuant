import torch
from MiniQuant.quant.per_channel import (
    symmetric_quantize_per_channel_int8,
    symmetric_dequantize_per_channel_int8
)

# 测试每个通道的scale是独立的
def test_each_channel_has_independent_scale():

    x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 2.0, 3.0, 4.0],
            [-0.01, 0.02, 0.03, 0.04],
        ]
    )

    q = symmetric_quantize_per_channel_int8(
        x,
        channel_dim=0,
    )

    expected = torch.tensor(
        [
            0.4 / 127.0,
            4.0 / 127.0,
            0.04 / 127.0,
        ]
    )

    assert torch.allclose(
        q.scale.squeeze(1),
        expected,
    )

# 测试反量化
def test_round_trip():

    torch.manual_seed(42)

    x = torch.randn(
        32,
        64,
    )

    q = symmetric_quantize_per_channel_int8(
        x,
        channel_dim=0,
    )

    x_hat = (
        symmetric_dequantize_per_channel_int8(q)
    )

    mse = torch.mean(
        (x - x_hat) ** 2
    )

    assert mse < 1e-3

# 测试全零张量
def test_zero_tensor():

    x = torch.zeros(
        8,
        16,
    )

    q = symmetric_quantize_per_channel_int8(
        x,
        channel_dim=0,
    )

    x_hat = (
        symmetric_dequantize_per_channel_int8(q)
    )

    assert torch.allclose(
        x,
        x_hat,
    )