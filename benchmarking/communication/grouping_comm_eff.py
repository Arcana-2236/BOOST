import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.distributed as dist


def benchmark_allreduce(op: Callable[[], None], warmup: int = 5, iters: int = 20) -> float:
    """Return average latency in microseconds for one benchmark iteration."""
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        op()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return (elapsed_ms * 1000.0) / iters


def format_results_table(rows: List[Dict[str, float]]) -> str:
    header = (
        f"{'Batch':<8} {'Workload':<10} {'Variant':<14} {'Calls':<8} "
        f"{'Payload (MB)':<14} {'Avg Latency (us)':<18}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{int(row['batch_size']):<8} {row['workload']:<10} {row['variant']:<14} "
            f"{int(row['calls']):<8} {row['payload_mb']:<14.2f} {row['latency_us']:<18.2f}"
        )
    return "\n".join(lines)


def format_speedup_table(rows: List[Dict[str, float]]) -> str:
    data: Dict[int, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        batch_size = int(row["batch_size"])
        workload = row["workload"]
        variant = row["variant"]
        data.setdefault(batch_size, {}).setdefault(workload, {})[variant] = row["latency_us"]

    header = f"{'Batch':<8} {'Workload':<10} {'Speedup (Non-grouped / Grouped)':<36}"
    sep = "-" * len(header)
    lines = [header, sep]
    for batch_size in sorted(data.keys()):
        for workload in ("Attn", "MLP"):
            non_grouped_us = data[batch_size][workload]["non-grouped"]
            grouped_us = data[batch_size][workload]["grouped"]
            speedup = non_grouped_us / grouped_us
            lines.append(f"{batch_size:<8} {workload:<10} {speedup:<36.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark grouped vs non-grouped communication and export summary tables."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Single batch size override (deprecated, use --batch-sizes).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Batch sizes to benchmark. Default: 1 4",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank_id = dist.get_rank()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # LLaMA 7B case
    batch_sizes = [args.batch_size] if args.batch_size is not None else args.batch_sizes
    batch_sizes = sorted({int(batch_size) for batch_size in batch_sizes})
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError(f"Batch sizes must be > 0, got: {batch_sizes}")

    seq_len, hidden_size, intermediate_size, rank, tp = 4096, 4096, 11008, 1024, 4
    warmup, iters = args.warmup, args.iters

    rows: List[Dict[str, float]] = []

    for batch_size in batch_sizes:
        # Non-grouped communication tensors
        non_grouped_activation_attn_q = torch.randn(seq_len, batch_size, rank, dtype=dtype, device=device)
        non_grouped_activation_attn_k = torch.randn(seq_len, batch_size, rank, dtype=dtype, device=device)
        non_grouped_activation_attn_v = torch.randn(seq_len, batch_size, rank, dtype=dtype, device=device)
        non_grouped_activation_mlp_0 = torch.randn(seq_len, batch_size, rank, dtype=dtype, device=device)
        non_grouped_activation_mlp_1 = torch.randn(seq_len, batch_size, rank, dtype=dtype, device=device)

        # Grouped communication tensors
        grouped_activation_attn = torch.randn(seq_len, batch_size, 3 * rank, dtype=dtype, device=device)
        grouped_activation_mlp = torch.randn(seq_len, batch_size, 2 * rank, dtype=dtype, device=device)

        def op_attn_non_grouped() -> None:
            dist.all_reduce(non_grouped_activation_attn_q, op=dist.ReduceOp.SUM)
            dist.all_reduce(non_grouped_activation_attn_k, op=dist.ReduceOp.SUM)
            dist.all_reduce(non_grouped_activation_attn_v, op=dist.ReduceOp.SUM)

        def op_attn_grouped() -> None:
            dist.all_reduce(grouped_activation_attn, op=dist.ReduceOp.SUM)

        def op_mlp_non_grouped() -> None:
            dist.all_reduce(non_grouped_activation_mlp_0, op=dist.ReduceOp.SUM)
            dist.all_reduce(non_grouped_activation_mlp_1, op=dist.ReduceOp.SUM)

        def op_mlp_grouped() -> None:
            dist.all_reduce(grouped_activation_mlp, op=dist.ReduceOp.SUM)

        payload_attn_mb = grouped_activation_attn.numel() * grouped_activation_attn.element_size() / (1024 ** 2)
        payload_mlp_mb = grouped_activation_mlp.numel() * grouped_activation_mlp.element_size() / (1024 ** 2)

        rows.append(
            {
                "batch_size": float(batch_size),
                "workload": "Attn",
                "variant": "non-grouped",
                "calls": 3.0,
                "payload_mb": payload_attn_mb,
                "latency_us": benchmark_allreduce(op_attn_non_grouped, warmup=warmup, iters=iters),
            }
        )
        rows.append(
            {
                "batch_size": float(batch_size),
                "workload": "Attn",
                "variant": "grouped",
                "calls": 1.0,
                "payload_mb": payload_attn_mb,
                "latency_us": benchmark_allreduce(op_attn_grouped, warmup=warmup, iters=iters),
            }
        )
        rows.append(
            {
                "batch_size": float(batch_size),
                "workload": "MLP",
                "variant": "non-grouped",
                "calls": 2.0,
                "payload_mb": payload_mlp_mb,
                "latency_us": benchmark_allreduce(op_mlp_non_grouped, warmup=warmup, iters=iters),
            }
        )
        rows.append(
            {
                "batch_size": float(batch_size),
                "workload": "MLP",
                "variant": "grouped",
                "calls": 1.0,
                "payload_mb": payload_mlp_mb,
                "latency_us": benchmark_allreduce(op_mlp_grouped, warmup=warmup, iters=iters),
            }
        )

    dist.barrier()
    if rank_id == 0:
        table_main = format_results_table(rows)
        table_speedup = format_speedup_table(rows)

        print("\nGrouping Communication Efficiency Table")
        print("=" * 80)
        print(table_main)
        print("\nGrouping Communication Speedup Table")
        print("=" * 80)
        print(table_speedup)

        output_path = Path(__file__).with_name("grouping_comm_eff_tables.txt")
        output_path.write_text(
            "\n".join(
                [
                    "Grouping Communication Efficiency Table",
                    "=" * 80,
                    f"seq_len={seq_len}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, rank={rank}, tp={tp}",
                    f"dtype={dtype}, warmup={warmup}, iters={iters}, batch_sizes={batch_sizes}",
                    "",
                    table_main,
                    "",
                    "Grouping Communication Speedup Table",
                    "=" * 80,
                    table_speedup,
                    "",
                ]
            )
        )
        print(f"\nSaved table results to: {output_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
