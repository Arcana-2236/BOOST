import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as torch_dist

from nanotron.nn.layer_norm import OnlineRMSNorm, TritonRMSNorm


def _reference_local_rmsnorm(
    x_local: torch.Tensor,
    gamma_local: torch.Tensor,
    eps: float,
    accum_dtype: torch.dtype = torch.float32,
):
    x_acc = x_local.to(accum_dtype)
    g_acc = gamma_local.to(accum_dtype)
    s_local = (x_acc * x_acc).sum(dim=-1, keepdim=True)
    d_local = x_local.shape[-1]
    rstd_local = torch.rsqrt(s_local / d_local + eps)
    y_local = (x_acc * rstd_local) * g_acc
    return y_local.to(x_local.dtype), s_local


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 5e-5, 5e-4
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    raise ValueError(dtype)


_TP_GROUP_CACHE = {}


def _tp_sizes_for_world(world_size: int):
    # Local-RMSNorm parity across no-shard (TP=1) and max shard (TP=world_size).
    sizes = [1]
    if world_size > 1:
        if world_size % 2 == 0:
            sizes.append(2)
        sizes.append(world_size)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(sizes))


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
def test_online_rmsnorm_matches_local_reference(dtype, shape, eps):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, e.g. torchrun --nproc_per_node=4 -m pytest -q -k online_rmsnorm")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for OnlineRMSNorm TP test")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    bsz, seq, hidden = shape
    atol, rtol = _tol(dtype)

    tested_tp_sizes = 0
    for tp_size in _tp_sizes_for_world(world_size):
        if world_size % tp_size != 0 or hidden % tp_size != 0:
            continue

        tp_group, tp_ranks, tp_rank, dp_rank = _get_tp_layout(tp_size=tp_size)
        hidden_local = hidden // tp_size
        start = tp_rank * hidden_local
        end = (tp_rank + 1) * hidden_local

        if tp_rank == 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(3000 + 31 * dp_rank + tp_size)
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

        online = OnlineRMSNorm(
            hidden_size=hidden_local,
            pg=tp_group,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            online.weight.copy_(gamma_local)

        y_local, s_local = online(x_local)
        y_ref, s_ref = _reference_local_rmsnorm(x_local, gamma_local, eps, accum_dtype=torch.float32)
        torch.testing.assert_close(y_local, y_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(s_local, s_ref, atol=1e-5, rtol=1e-5)

        (y_local * grad_local).sum().backward()
        dx_kernel = x_local.grad.detach().clone()
        dg_kernel = online.weight.grad.detach().clone()

        x_ref = x_local.detach().clone().requires_grad_(True)
        g_ref = gamma_local.detach().clone().requires_grad_(True)
        y_ref_bwd, _ = _reference_local_rmsnorm(x_ref, g_ref, eps, accum_dtype=torch.float32)
        (y_ref_bwd * grad_local).sum().backward()
        dx_ref = x_ref.grad
        dg_ref = g_ref.grad

        torch.testing.assert_close(dx_kernel, dx_ref, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dg_kernel, dg_ref, atol=atol * 2, rtol=rtol * 2)
        tested_tp_sizes += 1

    if tested_tp_sizes == 0:
        pytest.skip(f"No valid TP size in {_tp_sizes_for_world(world_size)} for hidden={hidden}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 1536), (2, 2, 2048)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_online_rmsnorm_vs_triton_local(dtype, shape, eps):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, e.g. torchrun --nproc_per_node=4 -m pytest -q -k online_rmsnorm")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for OnlineRMSNorm TP test")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    bsz, seq, hidden = shape
    atol, rtol = _tol(dtype)

    tested_tp_sizes = 0
    for tp_size in _tp_sizes_for_world(world_size):
        if world_size % tp_size != 0 or hidden % tp_size != 0:
            continue

        tp_group, tp_ranks, tp_rank, dp_rank = _get_tp_layout(tp_size=tp_size)
        hidden_local = hidden // tp_size
        start = tp_rank * hidden_local
        end = (tp_rank + 1) * hidden_local

        if tp_rank == 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(4000 + 29 * dp_rank + tp_size)
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

        online = OnlineRMSNorm(
            hidden_size=hidden_local,
            pg=tp_group,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        triton_local = TritonRMSNorm(hidden_local, eps=eps, device=device, dtype=dtype)
        with torch.no_grad():
            online.weight.copy_(gamma_local)
            triton_local.weight.copy_(gamma_local)

        y_online, s_online = online(x_local)
        y_triton = triton_local(x_local.detach().clone())
        y_ref, s_ref = _reference_local_rmsnorm(x_local, gamma_local, eps, accum_dtype=torch.float32)

        torch.testing.assert_close(y_online, y_triton, atol=atol, rtol=rtol)
        torch.testing.assert_close(y_online, y_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(s_online, s_ref, atol=1e-5, rtol=1e-5)

        (y_online * grad_local).sum().backward()
        dx_online = x_local.grad.detach().clone()
        dg_online = online.weight.grad.detach().clone()

        x_triton = x_local.detach().clone().requires_grad_(True)
        triton_bwd = TritonRMSNorm(hidden_local, eps=eps, device=device, dtype=dtype)
        with torch.no_grad():
            triton_bwd.weight.copy_(gamma_local)
        y_triton_bwd = triton_bwd(x_triton)
        (y_triton_bwd * grad_local).sum().backward()
        dx_triton = x_triton.grad.detach().clone()
        dg_triton = triton_bwd.weight.grad.detach().clone()

        x_ref = x_local.detach().clone().requires_grad_(True)
        g_ref = gamma_local.detach().clone().requires_grad_(True)
        y_ref_bwd, _ = _reference_local_rmsnorm(x_ref, g_ref, eps, accum_dtype=torch.float32)
        (y_ref_bwd * grad_local).sum().backward()
        dx_ref = x_ref.grad.detach().clone()
        dg_ref = g_ref.grad.detach().clone()

        torch.testing.assert_close(dx_online, dx_triton, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dg_online, dg_triton, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dx_online, dx_ref, atol=atol * 2, rtol=rtol * 2)
        torch.testing.assert_close(dg_online, dg_ref, atol=atol * 2, rtol=rtol * 2)
        tested_tp_sizes += 1

    if tested_tp_sizes == 0:
        pytest.skip(f"No valid TP size in {_tp_sizes_for_world(world_size)} for hidden={hidden}")
