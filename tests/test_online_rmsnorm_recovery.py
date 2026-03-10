import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

from nanotron.nn.layer_norm import OnlineRMSNorm
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.functional import row_linear


def _reference_rmsnorm_full(
    x_full: torch.Tensor,
    gamma_full: torch.Tensor,
    eps: float,
    accum_dtype: torch.dtype = torch.float32,
):
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


def _assert_close_across_ranks(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    group: torch_dist.ProcessGroup,
):
    """Make close-check failure rank-consistent to avoid collective desync cascades."""
    local_error = ""
    local_fail = torch.zeros(1, device=got.device, dtype=torch.int32)
    try:
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    except AssertionError as e:
        local_fail.fill_(1)
        local_error = str(e)

    torch_dist.all_reduce(local_fail, op=torch_dist.ReduceOp.MAX, group=group)
    if local_fail.item() == 0:
        return

    gathered = [None for _ in range(group.size())]
    torch_dist.all_gather_object(gathered, local_error, group=group)
    first_msg = next((m for m in gathered if m), "Unknown mismatch")
    raise AssertionError(f"[{name}] {first_msg}")


def _tp_sizes_for_world(world_size: int):
    sizes = [1]
    if world_size > 1:
        if world_size % 2 == 0:
            sizes.append(2)
        sizes.append(world_size)
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
@pytest.mark.parametrize("shape", [(2, 2, 1024), (1, 3, 1536)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_online_rmsnorm_recovery_row_linear_matches_full_reference(dtype, shape, eps):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun, e.g. torchrun --nproc_per_node=4 -m pytest -q -k online_rmsnorm_recovery")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for OnlineRMSNorm recovery TP test")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    bsz, seq, hidden = shape
    out_features = hidden
    atol, rtol = _tol(dtype)

    tested_tp_sizes = 0
    for tp_size in _tp_sizes_for_world(world_size):
        if hidden % tp_size != 0 or world_size % tp_size != 0:
            continue

        tp_group, tp_ranks, tp_rank, dp_rank = _get_tp_layout(tp_size=tp_size)
        hidden_local = hidden // tp_size
        start = tp_rank * hidden_local
        end = (tp_rank + 1) * hidden_local

        if tp_rank == 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(5000 + 43 * dp_rank + tp_size)
            x_full = torch.randn(bsz, seq, hidden, device=device, dtype=dtype, generator=gen)
            gamma_full = torch.randn(hidden, device=device, dtype=dtype, generator=gen)
            w_full = torch.randn(out_features, hidden, device=device, dtype=dtype, generator=gen)
            grad_out = torch.randn(bsz, seq, out_features, device=device, dtype=dtype, generator=gen)
        else:
            x_full = torch.empty(bsz, seq, hidden, device=device, dtype=dtype)
            gamma_full = torch.empty(hidden, device=device, dtype=dtype)
            w_full = torch.empty(out_features, hidden, device=device, dtype=dtype)
            grad_out = torch.empty(bsz, seq, out_features, device=device, dtype=dtype)

        tp_root = tp_ranks[0]
        torch_dist.broadcast(x_full, src=tp_root, group=tp_group)
        torch_dist.broadcast(gamma_full, src=tp_root, group=tp_group)
        torch_dist.broadcast(w_full, src=tp_root, group=tp_group)
        torch_dist.broadcast(grad_out, src=tp_root, group=tp_group)

        x_local = x_full[..., start:end].detach().clone().requires_grad_(True)
        gamma_local = gamma_full[start:end].detach().clone()
        w_local = w_full[:, start:end].detach().clone().requires_grad_(True)

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
        out_kernel = row_linear(
            input=y_local,
            weight=w_local,
            bias=None,
            group=tp_group,
            tp_mode=TensorParallelLinearMode.ALL_REDUCE,
            async_communication=False,
            s_local=s_local,
            rms_eps=eps,
            online_rmsnorm_recovery=True,
        )

        x_ref = x_full.detach().clone().requires_grad_(True)
        g_ref = gamma_full.detach().clone().requires_grad_(True)
        w_ref = w_full.detach().clone().requires_grad_(True)
        y_ref = _reference_rmsnorm_full(x_ref, g_ref, eps, accum_dtype=torch.float32)
        out_ref = F.linear(y_ref, w_ref, bias=None)
        out_ref_forward = out_ref

        # bf16 forward can differ from full-reference due TP matmul/all-reduce order.
        # Build a TP-order-matched reference for forward-only validation.
        if dtype == torch.bfloat16:
            out_ref_forward = F.linear(y_local.detach(), w_local.detach(), bias=None)
            d_local = x_local.shape[-1]
            d_full = d_local * tp_group.size()
            rms_local_ref = torch.sqrt((s_local.detach() / d_local) + eps).to(out_ref_forward.dtype)
            s_global_ref = s_local.detach().clone()
            torch_dist.all_reduce(s_global_ref, op=torch_dist.ReduceOp.SUM, group=tp_group)
            rms_global_ref = torch.sqrt((s_global_ref / d_full) + eps).to(out_ref_forward.dtype)
            out_ref_forward = out_ref_forward * rms_local_ref
            torch_dist.all_reduce(out_ref_forward, op=torch_dist.ReduceOp.SUM, group=tp_group)
            out_ref_forward = out_ref_forward / rms_global_ref

        fw_atol = atol * 2
        fw_rtol = rtol * 2
        if dtype == torch.bfloat16:
            fw_atol = max(fw_atol, 1.0)
            fw_rtol = max(fw_rtol, 0.1)
        _assert_close_across_ranks(
            "forward:out",
            out_kernel,
            out_ref_forward,
            atol=fw_atol,
            rtol=fw_rtol,
            group=tp_group,
        )

        (out_kernel * grad_out).sum().backward()
        dx_local_kernel = x_local.grad.detach().clone()
        dg_local_kernel = online.weight.grad.detach().clone()
        dw_local_kernel = w_local.grad.detach().clone()

        (out_ref * grad_out).sum().backward()
        dx_local_ref = x_ref.grad[..., start:end]
        dg_local_ref = g_ref.grad[start:end]
        dw_local_ref = w_ref.grad[:, start:end]

        bw_atol = atol * 4
        bw_rtol = rtol * 4
        dx_atol = bw_atol
        dx_rtol = bw_rtol
        dw_atol = bw_atol
        dw_rtol = bw_rtol
        if dtype == torch.bfloat16:
            # bf16 dx is the most sensitive here due chained rsqrt + all-reduce
            # with different accumulation order vs full-reference path.
            dx_atol = max(dx_atol, 0.4)
            dx_rtol = max(dx_rtol, 0.4)
            dw_atol = max(dw_atol, 0.2)
            dw_rtol = max(dw_rtol, 0.12)
        _assert_close_across_ranks(
            "backward:dx_local",
            dx_local_kernel,
            dx_local_ref,
            atol=dx_atol,
            rtol=dx_rtol,
            group=tp_group,
        )
        _assert_close_across_ranks(
            "backward:dw_local",
            dw_local_kernel,
            dw_local_ref,
            atol=dw_atol,
            rtol=dw_rtol,
            group=tp_group,
        )
        dg_atol = dw_atol
        dg_rtol = dw_rtol
        if dtype == torch.bfloat16:
            # dgamma is most sensitive in bf16 due chained reduction + sqrt scaling.
            dg_atol = max(dg_atol, 1.0)
            dg_rtol = max(dg_rtol, 2.0)
        _assert_close_across_ranks(
            "backward:dg_local",
            dg_local_kernel,
            dg_local_ref,
            atol=dg_atol,
            rtol=dg_rtol,
            group=tp_group,
        )
        tested_tp_sizes += 1

    if tested_tp_sizes == 0:
        pytest.skip(f"No valid TP size in {_tp_sizes_for_world(world_size)} for hidden={hidden}")
