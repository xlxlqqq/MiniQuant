import torch

from MiniQuant.layers.int4_linear import (
    WeightOnlyInt4Linear,
)


def test_weight_reconstruction():

    torch.manual_seed(42)

    linear = torch.nn.Linear(
        256,
        128,
    )

    qlinear = (
        WeightOnlyInt4Linear.from_float(
            linear,
            group_size=64,
        )
    )

    W_hat = (
        qlinear.dequantize_weight()
    )

    assert W_hat.shape == linear.weight.shape

    mse = torch.mean(
        (linear.weight - W_hat) ** 2
    )

    print("weight mse:", mse.item())

    assert mse < 0.1

def test_linear_output():

    torch.manual_seed(42)

    linear = torch.nn.Linear(
        256,
        128,
    )

    qlinear = (
        WeightOnlyInt4Linear.from_float(
            linear,
            group_size=64,
        )
    )

    x = torch.randn(
        8,
        256,
    )

    y_fp = linear(x)

    y_int4 = qlinear(x)

    assert y_fp.shape == y_int4.shape

    mse = torch.mean(
        (y_fp - y_int4) ** 2
    )

    print(
        "output mse:",
        mse.item(),
    )

    assert mse < 1.0