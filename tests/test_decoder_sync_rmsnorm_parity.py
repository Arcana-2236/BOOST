import os
import sys
from datetime import timedelta

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


def _finite_report(name: str, tensor: torch.Tensor) -> str:
    finite = torch.isfinite(tensor)
    num_finite = int(finite.sum().item())
    num_total = tensor.numel()
    msg = f"{name}: finite={num_finite}/{num_total}"
    if num_finite > 0:
        finite_vals = tensor[finite].float()
        msg += f", min={finite_vals.min().item():.4e}, max={finite_vals.max().item():.4e}, mean={finite_vals.mean().item():.4e}"
    return msg


def _forward_stage_outputs(layer, hidden_states: torch.Tensor, sequence_mask: torch.Tensor):
    residual = hidden_states
    ln = layer.input_layernorm(hidden_states)
    if isinstance(ln, tuple):
        ln = ln[0]

    attn_out = layer.attn(hidden_states=ln, sequence_mask=sequence_mask)["hidden_states"]
    h_after_attn_res = attn_out + residual

    residual2 = h_after_attn_res
    post_ln = layer.post_attention_layernorm(h_after_attn_res)
    if isinstance(post_ln, tuple):
        post_ln = post_ln[0]
    mlp_out = layer.mlp(hidden_states=post_ln)["hidden_states"]
    final_out = mlp_out + residual2
    return {
        "input_layernorm": ln,
        "attn_out": attn_out,
        "after_attn_residual": h_after_attn_res,
        "post_attention_layernorm": post_ln,
        "mlp_out": mlp_out,
        "final_out": final_out,
    }


def _copy_tp_shards_into_reference(tp_layer, ref_layer, group, world_size: int):
    tp_named = dict(tp_layer.named_parameters())
    with torch.no_grad():
        for name, p_ref in ref_layer.named_parameters():
            if name not in tp_named:
                continue
            p_tp = tp_named[name]
            if p_ref.shape == p_tp.shape:
                p_ref.copy_(p_tp)
                continue

            if p_ref.ndim != p_tp.ndim:
                raise RuntimeError(f"Parameter rank mismatch for {name}: tp={tuple(p_tp.shape)} ref={tuple(p_ref.shape)}")

            shard_dims = [
                d
                for d in range(p_ref.ndim)
                if p_ref.shape[d] == p_tp.shape[d] * world_size
                and all(p_ref.shape[k] == p_tp.shape[k] for k in range(p_ref.ndim) if k != d)
            ]
            if len(shard_dims) != 1:
                raise RuntimeError(
                    f"Cannot infer shard dim for {name}: tp={tuple(p_tp.shape)} ref={tuple(p_ref.shape)}"
                )
            shard_dim = shard_dims[0]
            gathered = [torch.empty_like(p_tp) for _ in range(world_size)]
            torch_dist.all_gather(gathered, p_tp, group=group)
            p_ref.copy_(torch.cat(gathered, dim=shard_dim))


def _init_decoder_params_for_parity(layer):
    # Some TP modules (e.g. BatchedTensorParallelColumnLinear) allocate with
    # torch.empty and rely on external initialization in training. For this
    # unit test, initialize explicitly to avoid NaNs from uninitialized memory.
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name.endswith("bias"):
                param.zero_()
            elif "layernorm.weight" in name:
                param.fill_(1.0)
            else:
                param.normal_(mean=0.0, std=0.02)


def _reference_view_for_tp_param(
    tp_param: torch.Tensor, ref_param: torch.Tensor, rank: int, world_size: int, name: str
) -> torch.Tensor:
    if tp_param.shape == ref_param.shape:
        return ref_param

    if tp_param.ndim != ref_param.ndim:
        raise RuntimeError(
            f"Parameter rank mismatch for {name}: tp={tuple(tp_param.shape)} ref={tuple(ref_param.shape)}"
        )

    shard_dims = [
        d
        for d in range(ref_param.ndim)
        if ref_param.shape[d] == tp_param.shape[d] * world_size
        and all(ref_param.shape[k] == tp_param.shape[k] for k in range(ref_param.ndim) if k != d)
    ]
    if len(shard_dims) != 1:
        raise RuntimeError(f"Cannot infer shard dim for {name}: tp={tuple(tp_param.shape)} ref={tuple(ref_param.shape)}")
    shard_dim = shard_dims[0]
    local = tp_param.shape[shard_dim]
    return ref_param.narrow(shard_dim, rank * local, local)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_decoder_layer_sync_rmsnorm_tp_matches_reference(dtype, eps):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, e.g. torchrun --nproc_per_node=2 -m pytest -q -k decoder_sync_rmsnorm")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        pytest.skip("This parity test is intended for TP world_size >= 2")

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    LlamaDecoderLayer, ColaLlamaConfig = _import_cola_modules()

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    hidden = 128
    if hidden % world_size != 0:
        pytest.skip(f"hidden={hidden} not divisible by world_size={world_size}")
    hidden_local = hidden // world_size
    start = rank * hidden_local
    end = (rank + 1) * hidden_local

    config = ColaLlamaConfig(
        hidden_size=hidden,
        intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=2,
        max_position_embeddings=64,
        attn_rank=32,
        mlp_rank=32,
        rms_norm_eps=eps,
        rmsnorm_type="sync",
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
    parallel_ref = ParallelismArgs(
        dp=1,
        pp=1,
        tp=1,
        expert_parallel_size=1,
        recompute_layer=False,
        tp_mode="ALL_REDUCE",
        tp_linear_async_communication=False,
    )

    # TP decoder under test.
    tp_layer = LlamaDecoderLayer(
        config=config,
        parallel_config=parallel_tp,
        tp_pg=torch_dist.group.WORLD,
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    tp_layer.train()
    _init_decoder_params_for_parity(tp_layer)

    # TP=1 reference decoder on each rank.
    single_rank_group = torch_dist.new_group(ranks=[rank])
    ref_layer = LlamaDecoderLayer(
        config=config,
        parallel_config=parallel_ref,
        tp_pg=single_rank_group,
        layer_idx=0,
    ).to(device=device, dtype=dtype)
    ref_layer.train()

    # Reconstruct full reference parameters from TP shards.
    _copy_tp_shards_into_reference(tp_layer, ref_layer, torch_dist.group.WORLD, world_size)
    tp_named = dict(tp_layer.named_parameters())
    ref_named = dict(ref_layer.named_parameters())

    # 1-step optimizer parity (AdamW) in addition to forward/backward parity.
    tp_optim = torch.optim.AdamW(tp_layer.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    ref_optim = torch.optim.AdamW(ref_layer.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    seq, batch = 8, 2
    if rank == 0:
        x_full = torch.randn(seq, batch, hidden, device=device, dtype=dtype)
        grad_full = torch.randn(seq, batch, hidden, device=device, dtype=dtype)
    else:
        x_full = torch.empty(seq, batch, hidden, device=device, dtype=dtype)
        grad_full = torch.empty(seq, batch, hidden, device=device, dtype=dtype)
    torch_dist.broadcast(x_full, src=0)
    torch_dist.broadcast(grad_full, src=0)

    sequence_mask = torch.ones(batch, seq, device=device, dtype=torch.bool)

    x_local = x_full[..., start:end].detach().clone().requires_grad_(True)
    out_tp_local = tp_layer(hidden_states=x_local, sequence_mask=sequence_mask)["hidden_states"]

    gathered_out = [torch.empty_like(out_tp_local) for _ in range(world_size)]
    torch_dist.all_gather(gathered_out, out_tp_local, group=torch_dist.group.WORLD)
    out_tp_full = torch.cat(gathered_out, dim=-1)

    x_ref = x_full.detach().clone().requires_grad_(True)
    out_ref_full = ref_layer(hidden_states=x_ref, sequence_mask=sequence_mask)["hidden_states"]

    tp_finite = bool(torch.isfinite(out_tp_full).all().item())
    ref_finite = bool(torch.isfinite(out_ref_full).all().item())
    if not (tp_finite and ref_finite):
        with torch.no_grad():
            tp_stages = _forward_stage_outputs(tp_layer, x_local.detach(), sequence_mask)
            ref_stages = _forward_stage_outputs(ref_layer, x_ref.detach(), sequence_mask)
        debug_lines = [f"rank={rank}, eps={eps}, dtype={dtype}"]
        debug_lines.append(_finite_report("tp_out_full", out_tp_full))
        debug_lines.append(_finite_report("ref_out_full", out_ref_full))
        for stage_name in (
            "input_layernorm",
            "attn_out",
            "after_attn_residual",
            "post_attention_layernorm",
            "mlp_out",
            "final_out",
        ):
            debug_lines.append(_finite_report(f"tp_stage:{stage_name}", tp_stages[stage_name]))
            debug_lines.append(_finite_report(f"ref_stage:{stage_name}", ref_stages[stage_name]))
        raise AssertionError("Non-finite values detected in decoder parity test.\n" + "\n".join(debug_lines))

    # Forward parity for full decoder output.
    torch.testing.assert_close(out_tp_full, out_ref_full, atol=6e-2, rtol=6e-2)

    tp_optim.zero_grad(set_to_none=True)
    ref_optim.zero_grad(set_to_none=True)
    grad_local = grad_full[..., start:end].contiguous()
    (out_tp_local * grad_local).sum().backward()
    (out_ref_full * grad_full).sum().backward()

    # Input gradient parity.
    gathered_dx = [torch.empty_like(x_local.grad) for _ in range(world_size)]
    torch_dist.all_gather(gathered_dx, x_local.grad, group=torch_dist.group.WORLD)
    dx_tp_full = torch.cat(gathered_dx, dim=-1)
    torch.testing.assert_close(dx_tp_full, x_ref.grad, atol=9e-2, rtol=9e-2)

    # Key parameter gradients parity: local RMSNorm shards must match reference slices.
    torch.testing.assert_close(
        tp_layer.input_layernorm.weight.grad,
        ref_layer.input_layernorm.weight.grad[start:end],
        atol=9e-2,
        rtol=9e-2,
    )
    torch.testing.assert_close(
        tp_layer.post_attention_layernorm.weight.grad,
        ref_layer.post_attention_layernorm.weight.grad[start:end],
        atol=9e-2,
        rtol=9e-2,
    )

    # One optimizer step, then compare updated TP params vs ref slices.
    tp_optim.step()
    ref_optim.step()

    check_param_names = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "attn.qkv_proj0.weight",
        "attn.o_proj1.weight",
        "mlp.gate_up_proj0.weight",
        "mlp.down_proj1.weight",
    )
    for name in check_param_names:
        if name not in tp_named or name not in ref_named:
            continue
        tp_param = tp_named[name]
        ref_view = _reference_view_for_tp_param(tp_param, ref_named[name], rank, world_size, name)
        torch.testing.assert_close(tp_param, ref_view, atol=1.2e-1, rtol=1.2e-1)

    # Post-step forward parity on same batch.
    with torch.no_grad():
        out_tp_local_post = tp_layer(hidden_states=x_local.detach(), sequence_mask=sequence_mask)["hidden_states"]
        gathered_out_post = [torch.empty_like(out_tp_local_post) for _ in range(world_size)]
        torch_dist.all_gather(gathered_out_post, out_tp_local_post, group=torch_dist.group.WORLD)
        out_tp_full_post = torch.cat(gathered_out_post, dim=-1)
        out_ref_full_post = ref_layer(hidden_states=x_ref.detach(), sequence_mask=sequence_mask)["hidden_states"]
    torch.testing.assert_close(out_tp_full_post, out_ref_full_post, atol=1.2e-1, rtol=1.2e-1)
