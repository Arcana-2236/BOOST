import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

from nanotron.nn.layer_norm import OnlineRMSNorm, SyncRMSNorm


def init_distributed() -> torch.device:
    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend=backend)
        else:
            # Best-effort local fallback for ad-hoc debugging without torchrun.
            try:
                if backend == "gloo":
                    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
                rendezvous_file = Path("/tmp") / f"rmsnorm_eff_rdzv_{os.getpid()}"
                if rendezvous_file.exists():
                    rendezvous_file.unlink()
                dist.init_process_group(
                    backend=backend,
                    rank=0,
                    world_size=1,
                    init_method=f"file://{rendezvous_file}",
                )
            except Exception as err:
                raise RuntimeError(
                    "Failed to initialize distributed in single-process mode. "
                    "Run with torchrun instead, e.g.:\n"
                    "  torchrun --nproc_per_node=4 evaluation/computation/rmsnorm_eff.py"
                ) from err

    return device


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def benchmark_forward(fn, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync_device(device)
    dist.barrier()

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            fn()
        end_event.record()
        sync_device(device)
        elapsed_ms = start_event.elapsed_time(end_event) / iters
    else:
        start_time = time.perf_counter()
        for _ in range(iters):
            fn()
        sync_device(device)
        end_time = time.perf_counter()
        elapsed_ms = ((end_time - start_time) * 1000.0) / iters

    dist.barrier()
    return float(elapsed_ms)


def reduce_max(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def format_main_table(rows: List[Dict[str, float]]) -> str:
    header = (
        f"{'Batch':<8} {'SyncRMSNorm (ms)':<18} {'OnlineRMSNorm (ms)':<20} "
        f"{'Speedup (Sync/Online)':<24}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{int(row['batch_size']):<8} {row['sync_ms']:<18.4f} {row['online_ms']:<20.4f} "
            f"{row['speedup']:<24.3f}"
        )
    return "\n".join(lines)


def format_throughput_table(rows: List[Dict[str, float]]) -> str:
    header = (
        f"{'Batch':<8} {'Sync Tokens/s':<16} {'Online Tokens/s':<18} "
        f"{'Seq Len':<10} {'Hidden Local':<14}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{int(row['batch_size']):<8} {row['sync_tokens_per_s']:<16.2f} "
            f"{row['online_tokens_per_s']:<18.2f} {int(row['seq_len']):<10} {int(row['hidden_local']):<14}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OnlineRMSNorm vs SyncRMSNorm runtime.")
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size override.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4], help="Batch sizes to benchmark.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096, help="Global hidden size before TP sharding.")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output txt path. Default: evaluation/computation/rmsnorm_eff_tables.txt",
    )
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.hidden_size % world_size != 0:
        raise ValueError(
            f"hidden_size ({args.hidden_size}) must be divisible by world size ({world_size})."
        )
    hidden_local = args.hidden_size // world_size

    batch_sizes = [args.batch_size] if args.batch_size is not None else args.batch_sizes
    batch_sizes = sorted({int(bs) for bs in batch_sizes})
    if any(bs <= 0 for bs in batch_sizes):
        raise ValueError(f"Batch sizes must be > 0, got: {batch_sizes}")

    sync_norm = SyncRMSNorm(
        hidden_size=args.hidden_size,
        pg=dist.group.WORLD,
        eps=args.eps,
        device=device,
        dtype=dtype,
    )
    online_norm = OnlineRMSNorm(
        hidden_size=hidden_local,
        pg=dist.group.WORLD,
        eps=args.eps,
        device=device,
        dtype=dtype,
    )

    # Ensure both norms use identical local gamma for fair runtime comparison.
    with torch.no_grad():
        gamma = torch.randn(hidden_local, device=device, dtype=dtype)
        sync_norm.weight.copy_(gamma)
        online_norm.weight.copy_(gamma)

    rows: List[Dict[str, float]] = []
    for batch_size in batch_sizes:
        x_local = torch.randn(args.seq_len, batch_size, hidden_local, dtype=dtype, device=device)

        sync_ms_local = benchmark_forward(
            lambda: sync_norm(x_local),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        online_ms_local = benchmark_forward(
            lambda: online_norm(x_local)[0],
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )

        sync_ms = reduce_max(sync_ms_local, device=device)
        online_ms = reduce_max(online_ms_local, device=device)

        tokens = float(args.seq_len * batch_size)
        rows.append(
            {
                "batch_size": float(batch_size),
                "sync_ms": sync_ms,
                "online_ms": online_ms,
                "speedup": sync_ms / max(online_ms, 1e-12),
                "sync_tokens_per_s": tokens / (sync_ms / 1000.0),
                "online_tokens_per_s": tokens / (online_ms / 1000.0),
                "seq_len": float(args.seq_len),
                "hidden_local": float(hidden_local),
            }
        )

    if rank == 0:
        table_main = format_main_table(rows)
        table_throughput = format_throughput_table(rows)

        print("\nRMSNorm Runtime Comparison (max-rank latency)")
        print("=" * 80)
        print(table_main)
        print("\nRMSNorm Throughput (per-rank tokens/s)")
        print("=" * 80)
        print(table_throughput)

        output_path = Path(args.output) if args.output else Path(__file__).with_name("rmsnorm_eff_tables.txt")
        output_path.write_text(
            "\n".join(
                [
                    "RMSNorm Runtime Comparison (max-rank latency)",
                    "=" * 80,
                    f"world_size={world_size}, dtype={dtype}, seq_len={args.seq_len}, hidden_size={args.hidden_size}, hidden_local={hidden_local}",
                    f"warmup={args.warmup}, iters={args.iters}, batch_sizes={batch_sizes}",
                    "",
                    table_main,
                    "",
                    "RMSNorm Throughput (per-rank tokens/s)",
                    "=" * 80,
                    table_throughput,
                    "",
                ]
            )
        )
        print(f"\nSaved table results to: {output_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
