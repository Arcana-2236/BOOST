import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as torch_dist

from nanotron.config import ParallelismArgs
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_all_reduce_sum


def _import_cola_modules():
    cola_dir = "/home/zhengyangwang/nanotron/examples/cola"
    if cola_dir not in sys.path:
        sys.path.insert(0, cola_dir)
    from cola_llama import LlamaDecoderLayer  # type: ignore
    from config_cola_llama import ColaLlamaConfig  # type: ignore

    return LlamaDecoderLayer, ColaLlamaConfig


def _init_dist_if_needed():
    if torch_dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    else:
        torch_dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29613",
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=2),
        )


def _init_decoder_params(layer, std: float):
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name.endswith("bias"):
                param.zero_()
            elif "layernorm.weight" in name:
                param.fill_(1.0)
            else:
                param.normal_(mean=0.0, std=std)


def _global_l2_norm(t: torch.Tensor, group) -> float:
    sq = (t.float() * t.float()).sum()
    if group.size() > 1:
        torch_dist.all_reduce(sq, op=torch_dist.ReduceOp.SUM, group=group)
    return float(torch.sqrt(sq).item())


def _grad_global_l2_norm(module: torch.nn.Module, group) -> float:
    total = torch.zeros((), device=next(module.parameters()).device, dtype=torch.float32)
    for p in module.parameters():
        if p.grad is not None:
            total = total + (p.grad.float() * p.grad.float()).sum()
    if group.size() > 1:
        torch_dist.all_reduce(total, op=torch_dist.ReduceOp.SUM, group=group)
    return float(torch.sqrt(total).item())


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compare_with_reference(current: List[Dict], reference: List[Dict], atol: float, rtol: float):
    if len(current) != len(reference):
        raise AssertionError(f"Step count mismatch: current={len(current)} reference={len(reference)}")

    keys = ["loss", "grad_norm_global"]
    sentinels = sorted(set(current[0]["sentinel_update_norms"].keys()) & set(reference[0]["sentinel_update_norms"].keys()))
    keys.extend([f"sentinel_update_norms.{k}" for k in sentinels])

    max_abs = {k: 0.0 for k in keys}
    max_rel = {k: 0.0 for k in keys}
    worst = {k: -1 for k in keys}

    for i, (cur, ref) in enumerate(zip(current, reference)):
        for k in keys:
            if k.startswith("sentinel_update_norms."):
                sk = k.split(".", 1)[1]
                a = float(cur["sentinel_update_norms"][sk])
                b = float(ref["sentinel_update_norms"][sk])
            else:
                a = float(cur[k])
                b = float(ref[k])
            absd = abs(a - b)
            reld = absd / max(abs(b), 1e-12)
            if absd > max_abs[k]:
                max_abs[k] = absd
                max_rel[k] = reld
                worst[k] = i

    failures = []
    for k in keys:
        if not (max_abs[k] <= atol + rtol * max(1.0, max_abs[k])):
            failures.append((k, max_abs[k], max_rel[k], worst[k]))

    print("== Mini parity diff summary ==")
    for k in keys:
        print(f"{k}: max_abs={max_abs[k]:.6e}, max_rel={max_rel[k]:.6e}, step={worst[k]}")
    if failures:
        details = ", ".join([f"{k}@{s}(abs={a:.3e},rel={r:.3e})" for k, a, r, s in failures])
        raise AssertionError(f"Parity comparison failed: {details}")


def main():
    parser = argparse.ArgumentParser(description="Mini end-to-end decoder parity run for SyncRMSNorm.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attn-rank", type=int, default=32)
    parser.add_argument("--mlp-rank", type=int, default=32)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--rmsnorm-type", type=str, default="sync", choices=["sync", "triton"])
    parser.add_argument("--metrics-file", type=str, default="")
    parser.add_argument("--reference-metrics-file", type=str, default="")
    parser.add_argument("--compare-atol", type=float, default=1e-2)
    parser.add_argument("--compare-rtol", type=float, default=1e-2)
    args = parser.parse_args()

    _init_dist_if_needed()
    rank = torch_dist.get_rank()
    world_size = torch_dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if args.hidden_size % world_size != 0:
        raise ValueError(f"hidden_size={args.hidden_size} must be divisible by world_size={world_size}")
    hidden_local = args.hidden_size // world_size
    start = rank * hidden_local
    end = (rank + 1) * hidden_local

    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        metrics_dir = repo_root / ".logging" / "mini-parity"
        metrics_path = metrics_dir / f"decoder_sync_rmsnorm_ws{world_size}_{args.dtype}_eps{args.eps}.jsonl"

    LlamaDecoderLayer, ColaLlamaConfig = _import_cola_modules()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = ColaLlamaConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        max_position_embeddings=128,
        attn_rank=args.attn_rank,
        mlp_rank=args.mlp_rank,
        rms_norm_eps=args.eps,
        rmsnorm_type=args.rmsnorm_type,
        rope_interleaved=False,
        vocab_size=32000,
        tie_word_embeddings=False,
    )
    parallel_tp = ParallelismArgs(
        dp=1,
        pp=1,
        tp=world_size,
        expert_parallel_size=1,
        recompute_layer=False,
        tp_mode="ALL_REDUCE",
        tp_linear_async_communication=False,
    )
    layer = LlamaDecoderLayer(config=config, parallel_config=parallel_tp, tp_pg=torch_dist.group.WORLD, layer_idx=0).to(
        device=device, dtype=dtype
    )
    layer.train()
    _init_decoder_params(layer, std=args.init_std)

    optimizer = torch.optim.AdamW(layer.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    if rank == 0:
        x_full = torch.randn(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype)
        target_full = torch.randn(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype)
    else:
        x_full = torch.empty(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype)
        target_full = torch.empty(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype)
    torch_dist.broadcast(x_full, src=0)
    torch_dist.broadcast(target_full, src=0)
    x_local_fixed = x_full[..., start:end].contiguous()
    target_local_fixed = target_full[..., start:end].contiguous()
    sequence_mask = torch.ones(args.batch_size, args.seq_len, device=device, dtype=torch.bool)

    sentinel_names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "attn.qkv_proj0.weight",
        "mlp.down_proj1.weight",
    ]
    named_params = dict(layer.named_parameters())
    sentinels = [n for n in sentinel_names if n in named_params]

    metrics_rows: List[Dict] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        before = {n: named_params[n].detach().clone() for n in sentinels}

        x_local = x_local_fixed.detach().clone().requires_grad_(True)
        out_local = layer(hidden_states=x_local, sequence_mask=sequence_mask)["hidden_states"]
        mse_local = ((out_local.float() - target_local_fixed.float()) ** 2).sum()
        if world_size > 1:
            mse_global = differentiable_all_reduce_sum(mse_local.clone(), group=torch_dist.group.WORLD)
        else:
            mse_global = mse_local
        loss = mse_global / (args.seq_len * args.batch_size * args.hidden_size)
        loss.backward()
        grad_norm = _grad_global_l2_norm(layer, torch_dist.group.WORLD)
        optimizer.step()

        sentinel_update_norms = {}
        for n in sentinels:
            delta = (named_params[n].detach() - before[n]).float()
            sentinel_update_norms[n] = _global_l2_norm(delta, torch_dist.group.WORLD)

        row = {
            "step": step,
            "world_size": world_size,
            "dtype": args.dtype,
            "eps": args.eps,
            "rmsnorm_type": args.rmsnorm_type,
            "loss": float(loss.detach().item()),
            "grad_norm_global": grad_norm,
            "sentinel_update_norms": sentinel_update_norms,
        }
        metrics_rows.append(row)
        if rank == 0:
            print(
                f"step={step:03d} loss={row['loss']:.6e} grad_norm={row['grad_norm_global']:.6e} "
                f"u[input_ln]={sentinel_update_norms.get('input_layernorm.weight', 0.0):.6e}"
            )

    if rank == 0:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            for row in metrics_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved metrics to {metrics_path}")

        if args.reference_metrics_file:
            ref_path = Path(args.reference_metrics_file)
            ref_rows = _load_jsonl(ref_path)
            _compare_with_reference(metrics_rows, ref_rows, atol=args.compare_atol, rtol=args.compare_rtol)
            print("Reference comparison PASSED")


if __name__ == "__main__":
    main()
