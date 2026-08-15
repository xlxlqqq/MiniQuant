# python -m MiniQuant.examples.compare_tensor_channel

import torch

from MiniQuant.quant.affine import (
    symmetric_quantize_int8,
    symmetric_dequantize_int8,
)

from MiniQuant.quant.per_channel import (
    symmetric_quantize_per_channel_int8,
    symmetric_dequantize_per_channel_int8,
)

from MiniQuant.utils.metrics import (
    mae,
    max_error,
    mse,
    relative_error,
)


def main():

    torch.manual_seed(42)

    out_features = 4096
    in_features = 4096

    W = torch.randn(
        out_features,
        in_features,
    )

    # -------------------------------------------------
    # Per-Tensor
    # -------------------------------------------------

    qt = symmetric_quantize_int8(W)

    W_tensor = (
        symmetric_dequantize_int8(qt)
    )

    # -------------------------------------------------
    # Per-Channel
    # -------------------------------------------------

    qc = symmetric_quantize_per_channel_int8(
        W,
        channel_dim=0,
    )

    W_channel = (
        symmetric_dequantize_per_channel_int8(qc)
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    print("=" * 70)
    print("Per-Tensor INT8 vs Per-Channel INT8")
    print("=" * 70)

    print()

    print("Per-Tensor")
    print("-" * 30)

    print(
        f"Scale shape     : "
        f"{tuple(qt.scale.shape)}"
    )

    print(
        f"MSE             : "
        f"{mse(W, W_tensor):.8e}"
    )

    print(
        f"MAE             : "
        f"{mae(W, W_tensor):.8e}"
    )

    print(
        f"Max Error       : "
        f"{max_error(W, W_tensor):.8e}"
    )

    print(
        f"Relative Error  : "
        f"{relative_error(W, W_tensor):.8e}"
    )

    print()

    print("Per-Channel")
    print("-" * 30)

    print(
        f"Scale shape     : "
        f"{tuple(qc.scale.shape)}"
    )

    print(
        f"MSE             : "
        f"{mse(W, W_channel):.8e}"
    )

    print(
        f"MAE             : "
        f"{mae(W, W_channel):.8e}"
    )

    print(
        f"Max Error       : "
        f"{max_error(W, W_channel):.8e}"
    )

    print(
        f"Relative Error  : "
        f"{relative_error(W, W_channel):.8e}"
    )


if __name__ == "__main__":
    main()