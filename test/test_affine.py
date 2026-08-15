import torch

from MiniQuant.quant.affine import (
    symmetric_quantize_int8, 
    symmetric_dequantize_int8, 
    QuantizedTensor,
)

# 测试量化数据类型是 int8
def test_int8_dtype():
    x = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    q = symmetric_quantize_int8(x)

    # assert = 如果这个条件不成立，就直接报错并停止程序
    assert q.q.dtype == torch.int8

# 测试量化后的数据范围在 int8 的范围内
def test_int8_range():
    x = torch.randn(1000)

    q = symmetric_quantize_int8(x)

    assert q.q.min() >= -127
    assert q.q.max() <= 127

# 测试零点是 0
def test_zero_point_is_zero():
    x = torch.randn(1000)

    q = symmetric_quantize_int8(x)

    assert q.zero_point.item() == 0

# 测试量化和反量化的误差
def test_round_trip_error():
    x = torch.randn(1000)

    q = symmetric_quantize_int8(x)

    x_hat = symmetric_dequantize_int8(q)
    mse = torch.mean((x - x_hat) ** 2)

    assert mse < 1e-3

# 测试全零张量的量化和反量化
def test_zero_tensor():
    x = torch.zeros(100)

    q = symmetric_quantize_int8(x)

    x_hat = symmetric_dequantize_int8(q)

    assert torch.allclose(
        x_hat,
        x,
    )

# 测试量化的 scale 是否正确
def test_scale():
    x = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    q = symmetric_quantize_int8(x)

    expected_scale = 2.0 / 127.0

    assert torch.allclose(
        q.scale,
        torch.tensor(expected_scale),
    )