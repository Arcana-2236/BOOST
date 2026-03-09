import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as torch_dist

from nanotron.nn.layer_norm import SyncRMSNorm, TritonRMSNorm


def _reference_rmsnorm_full(
    x_full: torch.Tensor, gamma_full: torch.Tensor, eps: float, accum_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    x_acc = x_full.to(accum_dtype)
    g_acc = gamma_full.to(accum_dtype)
    mean_sq = (x_acc * x_acc).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    y_full = (x_acc * rstd) * g_acc
    return y_full.to(x_full.dtype)


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 5e-5, 5e-4
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    raise ValueError(dtype)


_TP_GROUP_CACHE = {}


def _tp_sizes_for_world(world_size: int):
    # Match training parity checks across TP=1 and TP=world_size (DP adjusts accordingly).
    if world_size == 1:
        return [1]
    return [1, world_size]


def _get_tp_layout(tp_size: int):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size % tp_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must be divisible by tp_size={tp_size}")

    key = (world_size, tp_size)
    if key not in _TP_GROUP_CACHE:
        dp_size = world_size // tp_size
        groups = []
        for dp_rank in range(dp_size):
            ranks = tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
            groups.append((ranks, torch_dist.new_group(ranks=list(ranks))))
        _TP_GROUP_CACHE[key] = groups

    for dp_rank, (ranks, tp_group) in enumerate(_TP_GROUP_CACHE[key]):
        if rank in ranks:
            tp_rank = ranks.index(rank)
            return tp_group, ranks, tp_rank, dp_rank

    raise RuntimeError(f"Rank {rank} does not belong to any TP group for tp_size={tp_size}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 1536), (2, 2, 2048), (1, 3, 5504)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_sync_rmsnorm_matches_full_reference(dtype, shape, eps):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, e.g. torchrun --nproc_per_node=2 -m pytest -q -k sync_rmsnorm")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SyncRMSNorm TP test")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if shape[-1] % world_size != 0:
        pytest.skip(f"Hidden size {shape[-1]} is not divisible by world size {world_size}")

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    bsz, seq, hidden = shape
    atol, rtol = _tol(dtype)

    tested_tp_sizes = 0
    for tp_size in _tp_sizes_for_world(world_size):
        if hidden % tp_size != 0:
            continue

        tp_group, tp_ranks, tp_rank, dp_rank = _get_tp_layout(tp_size=tp_size)
        hidden_local = hidden // tp_size
        start = tp_rank * hidden_local
        end = (tp_rank + 1) * hidden_local

        if tp_rank == 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(1000 + dp_rank)
            x_full = torch.randn(bsz, seq, hidden, device=device, dtype=dtype, generator=gen)
            gamma_full = torch.randn(hidden, device=device, dtype=dtype, generator=gen)
            grad_full = torch.randn(bsz, seq, hidden, device=device, dtype=dtype, generator=gen)
        else:
            x_full = torch.empty(bsz, seq, hidden, device=device, dtype=dtype)
            gamma_full = torch.empty(hidden, device=device, dtype=dtype)
            grad_full = torch.empty(bsz, seq, hidden, device=device, dtype=dtype)

        tp_root = tp_ranks[0]
        torch_dist.broadcast(x_full, src=tp_root, group=tp_group)
        torch_dist.broadcast(gamma_full, src=tp_root, group=tp_group)
        torch_dist.broadcast(grad_full, src=tp_root, group=tp_group)

        x_local = x_full[..., start:end].detach().clone().requires_grad_(True)
        gamma_local = gamma_full[start:end].detach().clone()
        grad_local = grad_full[..., start:end].detach().clone()

        sync_rmsnorm = SyncRMSNorm(
            hidden_size=hidden,
            pg=tp_group,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            sync_rmsnorm.weight.copy_(gamma_local)

        y_local, s_local, s_global = sync_rmsnorm(x_local, return_stats=True)

        y_full_ref = _reference_rmsnorm_full(x_full, gamma_full, eps, accum_dtype=torch.float32)
        y_local_ref = y_full_ref[..., start:end]
        torch.testing.assert_close(y_local, y_local_ref, atol=atol, rtol=rtol)

        s_local_ref = (x_local.float() * x_local.float()).sum(dim=-1, keepdim=True)
        s_global_ref = s_local_ref.clone()
        torch_dist.all_reduce(s_global_ref, op=torch_dist.ReduceOp.SUM, group=tp_group)
        torch.testing.assert_close(s_local, s_local_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s_global, s_global_ref, atol=1e-5, rtol=1e-5)

        (y_local * grad_local).sum().backward()
        dx_local_kernel = x_local.grad.detach().clone()
        dg_local_kernel = sync_rmsnorm.weight.grad.detach().clone()

        x_full_ref = x_full.detach().clone().requires_grad_(True)
        gamma_full_ref = gamma_full.detach().clone().requires_grad_(True)
        y_full_ref_bwd = _reference_rmsnorm_full(x_full_ref, gamma_full_ref, eps, accum_dtype=torch.float32)
        (y_full_ref_bwd * grad_full).sum().backward()
        dx_local_ref = x_full_ref.grad[..., start:end]
        dg_local_ref = gamma_full_ref.grad[start:end]

        torch.testing.assert_close(dx_local_kernel, dx_local_ref, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dg_local_kernel, dg_local_ref, atol=atol * 2, rtol=rtol * 2)
        tested_tp_sizes += 1

    if tested_tp_sizes == 0:
        pytest.skip(f"No valid TP size in {_tp_sizes_for_world(world_size)} for hidden={hidden}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 1536), (2, 2, 2048)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_sync_rmsnorm_vs_triton_rmsnorm(dtype, shape, eps):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, e.g. torchrun --nproc_per_node=2 -m pytest -q -k sync_rmsnorm_vs_triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SyncRMSNorm/TritonRMSNorm comparison")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    bsz, seq, hidden = shape
    atol, rtol = _tol(dtype)

    tested_tp_sizes = 0
    for tp_size in _tp_sizes_for_world(world_size):
        if hidden % tp_size != 0:
            continue

        tp_group, tp_ranks, tp_rank, dp_rank = _get_tp_layout(tp_size=tp_size)
        hidden_local = hidden // tp_size
        start = tp_rank * hidden_local
        end = (tp_rank + 1) * hidden_local

        if tp_rank == 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(2000 + dp_rank)
            x_full = torch.randn(bsz, seq, hidden, device=device, dtype=dtype, generator=gen)
            gamma_full = torch.randn(hidden, device=device, dtype=dtype, generator=gen)
            grad_full = torch.randn(bsz, seq, hidden, device=device, dtype=dtype, generator=gen)
        else:
            x_full = torch.empty(bsz, seq, hidden, device=device, dtype=dtype)
            gamma_full = torch.empty(hidden, device=device, dtype=dtype)
            grad_full = torch.empty(bsz, seq, hidden, device=device, dtype=dtype)

        tp_root = tp_ranks[0]
        torch_dist.broadcast(x_full, src=tp_root, group=tp_group)
        torch_dist.broadcast(gamma_full, src=tp_root, group=tp_group)
        torch_dist.broadcast(grad_full, src=tp_root, group=tp_group)

        x_local = x_full[..., start:end].detach().clone().requires_grad_(True)
        gamma_local = gamma_full[start:end].detach().clone()
        grad_local = grad_full[..., start:end].detach().clone()

        sync_rmsnorm = SyncRMSNorm(
            hidden_size=hidden,
            pg=tp_group,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            sync_rmsnorm.weight.copy_(gamma_local)

        triton_full = TritonRMSNorm(hidden, eps=eps, device=device, dtype=dtype)
        with torch.no_grad():
            triton_full.weight.copy_(gamma_full)

        # Forward: Sync local shard vs Triton full reference slice.
        y_local_sync = sync_rmsnorm(x_local)
        y_full_triton = triton_full(x_full.detach().clone())
        y_local_triton = y_full_triton[..., start:end]

        # Also validate both against eager full reference.
        y_full_eager = _reference_rmsnorm_full(x_full, gamma_full, eps, accum_dtype=torch.float32)
        y_local_eager = y_full_eager[..., start:end]

        torch.testing.assert_close(y_local_sync, y_local_triton, atol=atol, rtol=rtol)
        torch.testing.assert_close(y_local_sync, y_local_eager, atol=atol, rtol=rtol)
        torch.testing.assert_close(y_local_triton, y_local_eager, atol=atol, rtol=rtol)

        # Backward: compare local grads from Sync vs Triton full reference.
        (y_local_sync * grad_local).sum().backward()
        dx_local_sync = x_local.grad.detach().clone()
        dg_local_sync = sync_rmsnorm.weight.grad.detach().clone()

        x_full_triton = x_full.detach().clone().requires_grad_(True)
        triton_full_bwd = TritonRMSNorm(hidden, eps=eps, device=device, dtype=dtype)
        with torch.no_grad():
            triton_full_bwd.weight.copy_(gamma_full)
        y_full_triton_bwd = triton_full_bwd(x_full_triton)
        (y_full_triton_bwd * grad_full).sum().backward()
        dx_local_triton = x_full_triton.grad[..., start:end]
        dg_local_triton = triton_full_bwd.weight.grad[start:end]

        # Eager backward reference for extra safety.
        x_full_eager = x_full.detach().clone().requires_grad_(True)
        gamma_full_eager = gamma_full.detach().clone().requires_grad_(True)
        y_full_eager_bwd = _reference_rmsnorm_full(x_full_eager, gamma_full_eager, eps, accum_dtype=torch.float32)
        (y_full_eager_bwd * grad_full).sum().backward()
        dx_local_eager = x_full_eager.grad[..., start:end]
        dg_local_eager = gamma_full_eager.grad[start:end]

        torch.testing.assert_close(dx_local_sync, dx_local_triton, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dg_local_sync, dg_local_triton, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dx_local_sync, dx_local_eager, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dg_local_sync, dg_local_eager, atol=atol * 2, rtol=rtol * 2)
        tested_tp_sizes += 1

    if tested_tp_sizes == 0:
        pytest.skip(f"No valid TP size in {_tp_sizes_for_world(world_size)} for hidden={hidden}")
