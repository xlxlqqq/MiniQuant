import torch

from MiniQuant.quant.groupwise import (
    symmetric_dequantize_groupwise_int8,
    symmetric_quantize_groupwise_int8,
)


def test_groupwise_shape():

    x = torch.randn(
        4,
        16,
    )

    q = symmetric_quantize_groupwise_int8(
        x,
        group_size=4,
    )

    assert q.q.shape == x.shape

    assert q.scale.shape == (
        4,
        4,
    )

def test_groupwise_dtype():

    x = torch.randn(
        4,
        16,
    )

    q = symmetric_quantize_groupwise_int8(
        x,
        group_size=4,
    )

    assert q.q.dtype == torch.int8

def test_groupwise_scale():

    x = torch.tensor(
        [
            [
                1.0, 2.0, 3.0, 4.0,
                0.1, 0.2, 0.3, 0.4,
            ]
        ]
    )

    q = symmetric_quantize_groupwise_int8(
        x,
        group_size=4,
    )

    expected = torch.tensor(
        [
            [
                4.0 / 127.0,
                0.4 / 127.0,
            ]
        ]
    )

    assert torch.allclose(
        q.scale,
        expected,
    )

def test_groupwise_round_trip():

    torch.manual_seed(42)

    x = torch.randn(
        32,
        128,
    )

    q = symmetric_quantize_groupwise_int8(
        x,
        group_size=32,
    )

    x_hat = (
        symmetric_dequantize_groupwise_int8(q)
    )

    mse = torch.mean(
        (x - x_hat) ** 2
    )

    assert mse < 1e-3

def test_invalid_group_size():

    x = torch.randn(
        4,
        10,
    )

    try:

        symmetric_quantize_groupwise_int8(
            x,
            group_size=4,
        )

    except ValueError:

        return

    raise AssertionError(
        "Expected ValueError"
    )