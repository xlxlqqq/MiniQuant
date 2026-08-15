import torch
import torch.nn.functional as F

from MiniQuant.layers.layer import QuantizedLinear

# 测试量化线性层的输出形状
def test_quantized_linear_shape():
    layer = QuantizedLinear(in_features=16, out_features=32)

    layer.quantize_weight_per_tensor()

    x = torch.randn(4, 16)

    y = layer(x)

    assert y.shape == (4, 32)

# 测试量化线性层的输出与浮点线性层的输出接近
def test_quantized_linear_close_to_fp():
    torch.manual_seed(42)

    layer = QuantizedLinear(in_features=16 * 128, out_features=32)

    x = torch.randn(32, 16 * 128)

    y_fp = F.linear(x, layer.weight, layer.bias)

    # 三种方法任意选择
    # layer.quantize_weight_per_tensor()
    layer.quantize_weight_groupwise()
    # layer.quantize_weight_per_channel()

    y_q = layer(x)

    mse = torch.mean((y_fp - y_q) ** 2)

    assert mse < 1e-3

# 测试量化权重的数据类型是 int8
def test_quantized_weight_dtype():
    layer = QuantizedLinear(in_features=16, out_features=32)

    layer.quantize_weight_per_tensor()

    assert layer.weight_q.q.dtype == torch.int8

# 测试需要在前向传播之前调用量化权重
def test_forward_requires_quantization():
    layer = QuantizedLinear(in_features=16, out_features=32)

    x = torch.randn(2, 16)

    try:
        layer(x)
    except RuntimeError:
        return
    
    raise AssertionError("Expected RuntimeError")