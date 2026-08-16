# python -m MiniQuant.examples.int4_storage
import argparse


def calculate_storage(num_weights: int, group_size: int):
    if num_weights <= 0:
        raise ValueError("num_weights must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    int4_bytes = num_weights * 4 / 8
    num_groups = num_weights // group_size
    scale_bytes = num_groups * 4
    total = int4_bytes + scale_bytes
    fp32_bytes = num_weights * 4

    saved_bytes = fp32_bytes - total
    saved_percent = (saved_bytes / fp32_bytes) * 100
    compression_ratio = fp32_bytes / total if total > 0 else float("inf")

    return {
        "num_weights": num_weights,
        "group_size": group_size,
        "int4_bytes": int4_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total,
        "fp32_bytes": fp32_bytes,
        "saved_bytes": saved_bytes,
        "saved_percent": saved_percent,
        "compression_ratio": compression_ratio,
    }


def format_mb(value):
    return f"{value / 1024**2:.2f} MB"

def report(num_weights, group_size):
    stats = calculate_storage(num_weights, group_size)
    print(
        f"Group={stats['group_size']:3d} | "
        f"FP32={format_mb(stats['fp32_bytes'])} | "
        f"INT4={format_mb(stats['total_bytes'])} | "
        f"Saved={stats['saved_bytes'] / 1024**2:.2f} MB | "
        f"Saved={stats['saved_percent']:.1f}% | "
        f"Compression={stats['compression_ratio']:.2f}x"
    )
    return stats


def build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate storage savings from FP32 to int4 with per-group scales."
    )
    parser.add_argument(
        "--num-weights",
        type=int,
        default=1_000_000,
        help="Number of weights to estimate (default: 1,000,000)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for the per-group scale (default: 128)",
    )
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Print a quick comparison for common group sizes (32, 64, 128, 256).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.all_groups:
        print("Storage comparison for common group sizes:\n")
        for size in (32, 64, 128, 256):
            report(args.num_weights, size)
        return

    print(f"Estimating storage for {args.num_weights} weights with group_size={args.group_size}\n")
    stats = report(args.num_weights, args.group_size)
    print()
    print(
        f"INT4 total storage: {stats['total_bytes'] / 1024**2:.2f} MB\n"
        f"FP32 total storage: {stats['fp32_bytes'] / 1024**2:.2f} MB\n"
        f"Saved storage: {stats['saved_bytes'] / 1024**2:.2f} MB ({stats['saved_percent']:.1f}%)"
    )


if __name__ == "__main__":
    main()