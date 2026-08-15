# python -m MiniQuant.examples.compare_groupwise
import torch

from MiniQuant.quant.affine import (
    symmetric_quantize_int8,
    symmetric_dequantize_int8,
)

from MiniQuant.quant.per_channel import (
    symmetric_quantize_per_channel_int8,
    symmetric_dequantize_per_channel_int8,
)

from MiniQuant.quant.groupwise import (
    symmetric_quantize_groupwise_int8,
    symmetric_dequantize_groupwise_int8,
)

from MiniQuant.utils.metrics import (
    mse,
    mae,
    max_error,
    relative_error,
)


def evaluate(
    name,
    x,
    x_hat,
):

    print(
        f"{name:<20} "
        f"MSE={mse(x, x_hat):.8e} "
        f"MAE={mae(x, x_hat):.8e} "
        f"Max={max_error(x, x_hat):.8e} "
        f"Rel={relative_error(x, x_hat):.8e}"
    )


def main():

    torch.manual_seed(42)

    out_features = 4096
    in_features = 4096

    W = torch.randn(
        out_features,
        in_features,
    )

    # Add channel-wise outliers.
    W[100] *= 10
    W[500] *= 20
    W[1000] *= 5

    print("=" * 100)
    print("INT8 Quantization Granularity Comparison")
    print("=" * 100)

    # ------------------------------------------------
    # Per-Tensor
    # ------------------------------------------------

    q = symmetric_quantize_int8(W)

    W_hat = symmetric_dequantize_int8(q)

    evaluate(
        "Per-Tensor",
        W,
        W_hat,
    )

    # ------------------------------------------------
    # Per-Channel
    # ------------------------------------------------

    q = symmetric_quantize_per_channel_int8(
        W,
        channel_dim=0,
    )

    W_hat = symmetric_dequantize_per_channel_int8(
        q
    )

    evaluate(
        "Per-Channel",
        W,
        W_hat,
    )

    # ------------------------------------------------
    # Group-wise
    # ------------------------------------------------

    for group_size in [
        256,
        128,
        64,
        32,
    ]:

        q = symmetric_quantize_groupwise_int8(
            W,
            group_size=group_size,
        )

        W_hat = (
            symmetric_dequantize_groupwise_int8(q)
        )

        evaluate(
            f"Group-{group_size}",
            W,
            W_hat,
        )


if __name__ == "__main__":
    main()