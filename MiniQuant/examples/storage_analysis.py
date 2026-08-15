# python -m MiniQuant.examples.storage_analysis

# 实际压缩率

def storage_bytes_groupwise(
    weight_shape,
    group_size,
    weight_bits=8,
    scale_bytes=4,
):
    out_features, in_features = weight_shape

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    num_weights = out_features * in_features
    weight_bytes = num_weights * weight_bits / 8

    # group_size 可能不整除 in_features；这里按向上取整计算实际组数
    num_groups = (in_features + group_size - 1) // group_size
    num_scales = out_features * num_groups
    scale_bytes_total = num_scales * scale_bytes

    return (
        weight_bytes,
        scale_bytes_total,
        weight_bytes + scale_bytes_total,
    )


def groupwise_compression_stats(
    weight_shape,
    group_size,
    weight_bits=4,
    scale_bytes=2,  # FP16 scale 更真实
):
    out_features, in_features = weight_shape

    num_weights = out_features * in_features
    fp16_bytes = num_weights * 2

    weight_bytes, scale_bytes_total, total_bytes = storage_bytes_groupwise(
        weight_shape,
        group_size,
        weight_bits,
        scale_bytes,
    )

    compression_ratio = fp16_bytes / total_bytes
    effective_bits = total_bytes * 8 / num_weights

    return {
        "fp16_MB": fp16_bytes / 1024**2,
        "quant_MB": total_bytes / 1024**2,
        "weight_MB": weight_bytes / 1024**2,
        "scale_MB": scale_bytes_total / 1024**2,
        "compression_ratio": compression_ratio,
        "effective_bits": effective_bits,
    }


def compare_group_sizes(
    weight_shape,
    group_sizes,
    weight_bits=4,
    scale_bytes=2,
):
    rows = []
    for group_size in group_sizes:
        stats = groupwise_compression_stats(
            weight_shape=weight_shape,
            group_size=group_size,
            weight_bits=weight_bits,
            scale_bytes=scale_bytes,
        )
        rows.append(
            {
                "group_size": group_size,
                "compression_ratio": stats["compression_ratio"],
                "effective_bits": stats["effective_bits"],
                "weight_MB": stats["weight_MB"],
                "scale_MB": stats["scale_MB"],
                "quant_MB": stats["quant_MB"],
            }
        )
    return rows


def main():
    weight_shape = (4096, 4096)
    group_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    out_features, in_features = weight_shape
    fp16_total_bytes = out_features * in_features * 2
    fp16_total_mb = fp16_total_bytes / 1024**2

    print("=" * 90)
    print("Group-wise Weight Compression Comparison")
    print(f"Weight shape: {weight_shape}")
    print(f"FP16 baseline: {fp16_total_mb:.3f} MB")
    print("=" * 90)
    print(f"{'group_size':>10} {'compression_ratio':>17} {'effective_bits':>15} {'quant_MB':>10} {'scale_MB':>10} {'weight_MB':>10}")
    print("-" * 90)

    for row in compare_group_sizes(
        weight_shape=weight_shape,
        group_sizes=group_sizes,
        weight_bits=4,
        scale_bytes=2,
    ):
        print(
            f"{row['group_size']:>10d} "
            f"{row['compression_ratio']:>17.3f}x "
            f"{row['effective_bits']:>15.3f} "
            f"{row['quant_MB']:>10.3f} "
            f"{row['scale_MB']:>10.3f} "
            f"{row['weight_MB']:>10.3f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
