import torch

from MiniQuant.quant.int4 import (
    quantize_groupwise_int4,
    dequantize_groupwise_int4,
)


def test_int4_range():

    x = torch.randn(
        32,
        128,
    )

    q = quantize_groupwise_int4(
        x,
        group_size=32,
    )

    assert q.q.min() >= -7
    assert q.q.max() <= 7


def test_scale_shape():

    x = torch.randn(
        32,
        128,
    )

    q = quantize_groupwise_int4(
        x,
        group_size=32,
    )

    assert q.scale.shape == (
        32,
        4,
    )


def test_round_trip():

    torch.manual_seed(42)

    x = torch.randn(
        32,
        128,
    )

    q = quantize_groupwise_int4(
        x,
        group_size=32,
    )

    x_hat = dequantize_groupwise_int4(q)

    mse = torch.mean(
        (x - x_hat) ** 2
    )

    assert mse < 0.1


def test_zero():

    x = torch.zeros(
        8,
        64,
    )

    q = quantize_groupwise_int4(
        x,
        group_size=32,
    )

    x_hat = dequantize_groupwise_int4(q)

    assert torch.allclose(
        x,
        x_hat,
    )


def test_invalid_group_size():

    x = torch.randn(
        4,
        10,
    )

    try:

        quantize_groupwise_int4(
            x,
            group_size=4,
        )

    except ValueError:

        return

    raise AssertionError(
        "Expected ValueError"
    )