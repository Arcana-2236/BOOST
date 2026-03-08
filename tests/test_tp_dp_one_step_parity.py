import os
import sys
from datetime import timedelta
from typing import Dict, Tuple

import pytest
import torch
import torch.distributed as torch_dist

from nanotron.config import ParallelismArgs


def _import_cola_modules():
    cola_dir = "/home/zhengyangwang/nanotron/examples/cola"
    if cola_dir not in sys.path:
        sys.path.insert(0, cola_dir)
    from cola_llama import LlamaDecoderLayer  # type: ignore
    from config_cola_llama import ColaLlamaConfig  # type: ignore

    return LlamaDecoderLayer, ColaLlamaConfig


def _infer_shard_dim(local_shape: Tuple[int, ...], full_shape: Tuple[int, ...], tp_size: int) -> int:
    shard_dims = [
        d
        for d in range(len(full_shape))
        if full_shape[d] == local_shape[d] * tp_size
        and all(full_shape[k] == local_shape[k] for k in range(len(full_shape)) if k != d)
    ]
    if len(shard_dims) != 1:
        raise RuntimeError(
            f"Cannot infer TP shard dim: local={local_shape}, full={full_shape}, tp={tp_size}"
        )
    return shard_dims[0]


def _tensor_checksum(x: torch.Tensor) -> float:
    return float(x.detach().float().sum().item())


def _reconstruct_full_from_tp(local: torch.Tensor, full_shape: Tuple[int, ...], tp_group) -> torch.Tensor:
    tp_size = tp_group.size()
    if tuple(local.shape) == tuple(full_shape):
        return local.detach().clone()
    shard_dim = _infer_shard_dim(tuple(local.shape), tuple(full_shape), tp_size)
    gathered = [torch.empty_like(local) for _ in range(tp_size)]
    torch_dist.all_gather(gathered, local, group=tp_group)
    return torch.cat(gathered, dim=shard_dim)


def _copy_full_to_tp_shard(param_tp: torch.Tensor, param_full: torch.Tensor, tp_rank: int, tp_size: int) -> None:
    if tuple(param_tp.shape) == tuple(param_full.shape):
        param_tp.copy_(param_full.to(device=param_tp.device, dtype=param_tp.dtype))
        return
    shard_dim = _infer_shard_dim(tuple(param_tp.shape), tuple(param_full.shape), tp_size)
    local = param_tp.shape[shard_dim]
    shard = param_full.narrow(shard_dim, tp_rank * local, local)
    param_tp.copy_(shard.to(device=param_tp.device, dtype=param_tp.dtype))


def _max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_one_step_parity(dtype):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun --standalone --nproc_per_node=4 -m pytest -q -k one_step_parity")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        pytest.skip("This harness expects WORLD_SIZE=4.")

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    LlamaDecoderLayer, ColaLlamaConfig = _import_cola_modules()

    # Mode A: TP=1, DP=4
    dp_group_a = torch_dist.group.WORLD
    tp_group_a = torch_dist.new_group(ranks=[rank])

    # Mode B: TP=4, DP=1
    tp_group_b = torch_dist.group.WORLD
    dp_group_b = torch_dist.new_group(ranks=[rank])
    _ = dp_group_b  # not used further; created for symmetry.

    hidden = 256
    seq = 8
    batch = 8
    config = ColaLlamaConfig(
        hidden_size=hidden,
        intermediate_size=768,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        max_position_embeddings=64,
        attn_rank=64,
        mlp_rank=64,
        rms_norm_eps=1e-6,
        rmsnorm_type="sync",
        rope_interleaved=False,
        vocab_size=32000,
        tie_word_embeddings=False,
    )
    parallel_a = ParallelismArgs(
        dp=4,
        pp=1,
        tp=1,
        expert_parallel_size=1,
        recompute_layer=False,
        tp_mode="ALL_REDUCE",
        tp_linear_async_communication=False,
    )
    parallel_b = ParallelismArgs(
        dp=1,
        pp=1,
        tp=4,
        expert_parallel_size=1,
        recompute_layer=False,
        tp_mode="ALL_REDUCE",
        tp_linear_async_communication=False,
    )

    # Use a non-final decoder layer so TP output is still sharded.
    layer_idx = 0
    model_a = LlamaDecoderLayer(
        config=config, parallel_config=parallel_a, tp_pg=tp_group_a, layer_idx=layer_idx
    ).to(device=device, dtype=dtype)
    model_b = LlamaDecoderLayer(
        config=config, parallel_config=parallel_b, tp_pg=tp_group_b, layer_idx=layer_idx
    ).to(device=device, dtype=dtype)
    model_a.train()
    model_b.train()

    # Deterministic init on rank0, then broadcast full params for mode A.
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    with torch.no_grad():
        for name, p in model_a.named_parameters():
            if rank == 0:
                if name.endswith("bias"):
                    p.zero_()
                elif "layernorm.weight" in name:
                    p.fill_(1.0)
                else:
                    p.normal_(mean=0.0, std=0.02)
            torch_dist.broadcast(p, src=0, group=torch_dist.group.WORLD)

    # Copy/shard full params from mode A into mode B so starts are exactly matched.
    named_a = dict(model_a.named_parameters())
    named_b = dict(model_b.named_parameters())
    tp_rank_b = torch_dist.get_rank(group=tp_group_b)
    tp_size_b = tp_group_b.size()
    with torch.no_grad():
        for name, p_b in named_b.items():
            p_a = named_a[name]
            _copy_full_to_tp_shard(p_b, p_a, tp_rank=tp_rank_b, tp_size=tp_size_b)

    stage = "initial weights"
    # Initial full-weight checksum equality: A full vs reconstructed B full.
    init_max_abs = 0.0
    if rank == 0:
        print("[one_step_parity] Checking initial full-weight identity...")
    with torch.no_grad():
        for name, p_a in named_a.items():
            p_b = named_b[name]
            p_b_full = _reconstruct_full_from_tp(p_b, tuple(p_a.shape), tp_group_b)
            diff = (p_a.detach() - p_b_full).abs().max().item()
            init_max_abs = max(init_max_abs, float(diff))
    init_max_abs_t = torch.tensor(init_max_abs, device=device)
    torch_dist.all_reduce(init_max_abs_t, op=torch_dist.ReduceOp.MAX)
    init_max_abs_global = float(init_max_abs_t.item())

    # Input data: generate once on rank0 and broadcast to all ranks.
    stage = "input batch"
    if rank == 0:
        torch.manual_seed(777)
        x_full = torch.randn(seq, batch, hidden, device=device, dtype=dtype)
        target_full = torch.randn(seq, batch, hidden, device=device, dtype=dtype)
    else:
        x_full = torch.empty(seq, batch, hidden, device=device, dtype=dtype)
        target_full = torch.empty(seq, batch, hidden, device=device, dtype=dtype)
    torch_dist.broadcast(x_full, src=0)
    torch_dist.broadcast(target_full, src=0)
    input_checksum = torch.tensor(
        [_tensor_checksum(x_full), _tensor_checksum(target_full)], device=device, dtype=torch.float64
    )
    input_checksum_min = input_checksum.clone()
    input_checksum_max = input_checksum.clone()
    torch_dist.all_reduce(input_checksum_min, op=torch_dist.ReduceOp.MIN)
    torch_dist.all_reduce(input_checksum_max, op=torch_dist.ReduceOp.MAX)
    input_checksum_span = float((input_checksum_max - input_checksum_min).abs().max().item())

    # Mode A uses the same batch on all DP ranks, then DP grad average.
    x_a = x_full.detach().clone().requires_grad_(True)
    # Mode B TP-shards hidden dim.
    hidden_local = hidden // tp_size_b
    start = tp_rank_b * hidden_local
    end = (tp_rank_b + 1) * hidden_local
    x_b_local = x_full[..., start:end].detach().clone().requires_grad_(True)
    target_b_local = target_full[..., start:end].detach()

    sequence_mask = torch.ones(batch, seq, device=device, dtype=torch.bool)

    optim_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    optim_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    optim_a.zero_grad(set_to_none=True)
    optim_b.zero_grad(set_to_none=True)

    stage = "forward outputs"
    out_a = model_a(hidden_states=x_a, sequence_mask=sequence_mask)["hidden_states"]
    out_b_local = model_b(hidden_states=x_b_local, sequence_mask=sequence_mask)["hidden_states"]
    out_b_full = _reconstruct_full_from_tp(out_b_local, tuple(out_a.shape), tp_group_b)

    out_diff_abs, out_diff_rel = _max_abs_rel(out_b_full.float(), out_a.float())
    loss_a = ((out_a - target_full) ** 2).mean()
    # Local contribution to full MSE mean, no TP autograd collective required.
    loss_b_local = ((out_b_local - target_b_local) ** 2).sum() / (seq * batch * hidden)
    loss_b_value = loss_b_local.detach().clone()
    torch_dist.all_reduce(loss_b_value, op=torch_dist.ReduceOp.SUM, group=tp_group_b)
    loss_b = loss_b_value

    stage = "grads"
    loss_a.backward()
    loss_b_local.backward()

    # Emulate DP=4 sync in mode A.
    for p in model_a.parameters():
        if p.grad is not None:
            torch_dist.all_reduce(p.grad, op=torch_dist.ReduceOp.AVG, group=dp_group_a)

    grad_max_abs = 0.0
    with torch.no_grad():
        for name, p_a in named_a.items():
            p_b = named_b[name]
            if p_a.grad is None or p_b.grad is None:
                continue
            g_b_full = _reconstruct_full_from_tp(p_b.grad, tuple(p_a.grad.shape), tp_group_b)
            grad_max_abs = max(grad_max_abs, float((p_a.grad.float() - g_b_full.float()).abs().max().item()))
    grad_max_abs_t = torch.tensor(grad_max_abs, device=device)
    torch_dist.all_reduce(grad_max_abs_t, op=torch_dist.ReduceOp.MAX)
    grad_max_abs_global = float(grad_max_abs_t.item())

    stage = "post-step weights"
    optim_a.step()
    optim_b.step()

    post_param_diffs: Dict[str, float] = {}
    with torch.no_grad():
        for name, p_a in named_a.items():
            p_b = named_b[name]
            p_b_full = _reconstruct_full_from_tp(p_b, tuple(p_a.shape), tp_group_b)
            post_param_diffs[name] = float((p_a.float() - p_b_full.float()).abs().max().item())
    post_max_abs = max(post_param_diffs.values()) if len(post_param_diffs) > 0 else 0.0
    post_max_abs_t = torch.tensor(post_max_abs, device=device)
    torch_dist.all_reduce(post_max_abs_t, op=torch_dist.ReduceOp.MAX)
    post_max_abs_global = float(post_max_abs_t.item())

    if rank == 0:
        print(f"[one_step_parity] loss_a={loss_a.item():.8f} loss_b={loss_b.item():.8f}")
        print(
            "[one_step_parity] forward diff: "
            f"max_abs={out_diff_abs:.8e}, max_rel={out_diff_rel:.8e}"
        )
        print(f"[one_step_parity] init weight max_abs={init_max_abs_global:.8e}")
        print(f"[one_step_parity] input checksum span={input_checksum_span:.8e}")
        print(f"[one_step_parity] grad max_abs={grad_max_abs_global:.8e}")
        print(f"[one_step_parity] post-step weight max_abs={post_max_abs_global:.8e}")
        major = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "attn.qkv_proj0.weight",
            "attn.o_proj1.weight",
            "mlp.gate_up_proj0.weight",
            "mlp.down_proj1.weight",
        ]
        for name in major:
            if name in post_param_diffs:
                print(f"[one_step_parity] post-step param diff {name}: {post_param_diffs[name]:.8e}")

    # Run kernel in bf16, but compare in fp32 for stable parity checks.
    try:
        assert init_max_abs_global < 1e-7, f"Initial weights diverged: max_abs={init_max_abs_global}"
        assert input_checksum_span < 1e-7, f"Input batch diverged: checksum_span={input_checksum_span}"
        torch.testing.assert_close(out_b_full.float(), out_a.float(), atol=3e-3, rtol=3e-3)
        assert grad_max_abs_global < 5e-3, f"Gradients diverged: max_abs={grad_max_abs_global}"
        assert post_max_abs_global < 5e-3, f"Post-step weights diverged: max_abs={post_max_abs_global}"
        if rank == 0:
            print("[one_step_parity] PASS")
    except Exception as exc:
        if rank == 0:
            print(f"[one_step_parity] FAIL at stage: {stage}")
            print(f"[one_step_parity] error: {exc}")
        raise
