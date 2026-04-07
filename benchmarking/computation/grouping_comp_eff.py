import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch


def non_grouped_qkv_gemm(
    x: torch.Tensor,
    q0_weight: torch.Tensor,
    k0_weight: torch.Tensor,
    v0_weight: torch.Tensor,
    q1_weight: torch.Tensor,
    k1_weight: torch.Tensor,
    v1_weight: torch.Tensor,
):
    lr_q = torch.matmul(x, q0_weight)
    lr_k = torch.matmul(x, k0_weight)
    lr_v = torch.matmul(x, v0_weight)
    q_result = torch.matmul(lr_q, q1_weight)
    k_result = torch.matmul(lr_k, k1_weight)
    v_result = torch.matmul(lr_v, v1_weight)
    return q_result, k_result, v_result


def grouped_qkv_gemm(x_states: torch.Tensor, grouped_down_weight: torch.Tensor, grouped_up_weight: torch.Tensor):
    seq_len, batch_size, _ = x_states.shape
    qkv_states = torch.matmul(x_states, grouped_down_weight)
    qkv = qkv_states.view(seq_len, batch_size, 3, -1)
    qkv_result = torch.einsum("s b c r, c r d -> s b c d", qkv, grouped_up_weight)
    qkv_result = qkv_result.permute(2, 0, 1, 3)
    return qkv_result


def non_grouped_mlp_gemm(
    x: torch.Tensor,
    gate0_weight: torch.Tensor,
    up0_weight: torch.Tensor,
    gate1_weight: torch.Tensor,
    up1_weight: torch.Tensor,
):
    lr_gate = torch.matmul(x, gate0_weight)
    lr_up = torch.matmul(x, up0_weight)
    gate_result = torch.matmul(lr_gate, gate1_weight)
    up_result = torch.matmul(lr_up, up1_weight)
    return gate_result, up_result


def grouped_mlp_gemm(x_states: torch.Tensor, grouped_down_weight: torch.Tensor, grouped_up_weight: torch.Tensor):
    seq_len, batch_size, _ = x_states.shape
    mlp_states = torch.matmul(x_states, grouped_down_weight)
    mlp = mlp_states.view(seq_len, batch_size, 2, -1)
    mlp_result = torch.einsum("s b c r, c r d -> s b c d", mlp, grouped_up_weight)
    mlp_result = mlp_result.permute(2, 0, 1, 3)
    return mlp_result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def benchmark(op: Callable[[], object], device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        op()
    _sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        op()
    _sync(device)
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def format_table(rows: List[Dict[str, object]]) -> str:
    header = (
        f"{'Batch':<8} {'Workload':<10} {'Variant':<14} {'Avg Latency (ms)':<18} "
        f"{'Estimated TFLOPs':<18} {'Estimated TFLOPs/s':<18}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['batch_size']:<8} {r['workload']:<10} {r['variant']:<14} {r['lat_ms']:<18.3f} "
            f"{r['tflops']:<18.3f} {r['tflops_per_s']:<18.3f}"
        )
    return "\n".join(lines)


def format_speedup_table(rows: List[Dict[str, object]]) -> str:
    by_workload: Dict[int, Dict[str, Dict[str, float]]] = {}
    for r in rows:
        bs = int(r["batch_size"])
        wl = str(r["workload"])
        by_workload.setdefault(bs, {})
        by_workload[bs].setdefault(wl, {})
        by_workload[bs][wl][str(r["variant"])] = float(r["lat_ms"])

    header = f"{'Batch':<8} {'Workload':<10} {'Speedup (Non-grouped / Grouped)':<36}"
    sep = "-" * len(header)
    lines = [header, sep]
    for bs in sorted(by_workload.keys()):
        for workload in ("QKV", "MLP"):
            ng = by_workload[bs][workload]["non-grouped"]
            g = by_workload[bs][workload]["grouped"]
            lines.append(f"{bs:<8} {workload:<10} {ng / g:<36.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare grouped vs non-grouped QKV/MLP GEMMs and export tables.")
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size override (deprecated, use --batch-sizes).")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4], help="Batch sizes to benchmark. Default: 1 4")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--tp-world-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output txt path. Default: evaluation/computation/grouping_eff_tables.txt",
    )
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available. Running on CPU.")

    batch_sizes = [args.batch_size] if args.batch_size is not None else args.batch_sizes
    batch_sizes = sorted({int(bs) for bs in batch_sizes})
    if any(bs <= 0 for bs in batch_sizes):
        raise ValueError(f"Batch sizes must be > 0, got: {batch_sizes}")

    seq_len = args.seq_len
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    rank = args.rank
    tp_world_size = args.tp_world_size
    d_local = hidden_size // tp_world_size
    dff_local = intermediate_size // tp_world_size

    # QKV weights
    qkv_grouped_down = torch.randn(d_local, 3 * rank, dtype=dtype, device=device)
    qkv_grouped_up = torch.randn(3, rank, d_local, dtype=dtype, device=device)
    q0_weight, k0_weight, v0_weight = torch.split(qkv_grouped_down, rank, dim=1)
    q1_weight, k1_weight, v1_weight = qkv_grouped_up[0], qkv_grouped_up[1], qkv_grouped_up[2]

    # MLP weights
    mlp_grouped_down = torch.randn(d_local, 2 * rank, dtype=dtype, device=device)
    mlp_grouped_up = torch.randn(2, rank, dff_local, dtype=dtype, device=device)
    gate0_weight, up0_weight = torch.split(mlp_grouped_down, rank, dim=1)
    gate1_weight, up1_weight = mlp_grouped_up[0], mlp_grouped_up[1]

    rows: List[Dict[str, object]] = []
    print(
        f"Device: {device}, dtype: {dtype}, warmup={args.warmup}, iters={args.iters}, "
        f"batch_sizes={batch_sizes}"
    )

    for batch_size in batch_sizes:
        m = seq_len * batch_size
        x_states = torch.randn(seq_len, batch_size, d_local, dtype=dtype, device=device)

        ops = [
            (
                "QKV",
                "non-grouped",
                lambda: non_grouped_qkv_gemm(
                    x_states, q0_weight, k0_weight, v0_weight, q1_weight, k1_weight, v1_weight
                ),
                12 * m * d_local * rank,
            ),
            (
                "QKV",
                "grouped",
                lambda: grouped_qkv_gemm(x_states, qkv_grouped_down, qkv_grouped_up),
                12 * m * d_local * rank,
            ),
            (
                "MLP",
                "non-grouped",
                lambda: non_grouped_mlp_gemm(x_states, gate0_weight, up0_weight, gate1_weight, up1_weight),
                4 * m * d_local * rank + 4 * m * rank * dff_local,
            ),
            (
                "MLP",
                "grouped",
                lambda: grouped_mlp_gemm(x_states, mlp_grouped_down, mlp_grouped_up),
                4 * m * d_local * rank + 4 * m * rank * dff_local,
            ),
        ]

        for workload, variant, op, flops in ops:
            lat_ms = benchmark(op, device=device, warmup=args.warmup, iters=args.iters)
            tflops = flops / 1e12
            tflops_per_s = (flops / (lat_ms / 1000.0)) / 1e12
            rows.append(
                {
                    "batch_size": batch_size,
                    "workload": workload,
                    "variant": variant,
                    "lat_ms": lat_ms,
                    "tflops": tflops,
                    "tflops_per_s": tflops_per_s,
                }
            )

    table_main = format_table(rows)
    table_speedup = format_speedup_table(rows)
    print("\n=== Grouping Efficiency Summary ===")
    print(table_main)
    print("\n=== Speedup Summary ===")
    print(table_speedup)

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).with_name("grouping_eff_tables.txt")
    )
    output_path.write_text(
        "\n".join(
            [
                "Grouping Efficiency Summary",
                "=" * 80,
                f"Device: {device}",
                f"dtype: {dtype}",
                f"batch_sizes={batch_sizes}, seq_len={seq_len}, hidden_size={hidden_size}, "
                f"intermediate_size={intermediate_size}, rank={rank}, tp_world_size={tp_world_size}",
                "",
                table_main,
                "",
                table_speedup,
                "",
            ]
        )
    )
    print(f"\nSaved table results to: {output_path}")


if __name__ == "__main__":
    main()
