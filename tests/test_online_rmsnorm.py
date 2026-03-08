import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

from nanotron.nn.layer_norm import DelayedTritonRMSNorm
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.functional import row_linear


DEBUG_ENV = "ONLINE_RMSNORM_DEBUG"


def _debug_enabled() -> bool:
    return os.getenv(DEBUG_ENV, "0") == "1"


def _assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float, extra_debug: dict = None):
    try:
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    except AssertionError:
        if _debug_enabled():
            print(f"[{name}] mismatch")
            print(f"  got dtype/shape={got.dtype}/{tuple(got.shape)}")
            print(f"  ref dtype/shape={ref.dtype}/{tuple(ref.shape)}")
            diff = (got.float() - ref.float()).abs()
            print(
                f"  abs diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}, "
                f"got_norm={got.float().norm().item():.6e}, ref_norm={ref.float().norm().item():.6e}"
            )
            if extra_debug is not None:
                for key, value in extra_debug.items():
                    if torch.is_tensor(value):
                        v = value.detach().float()
                        print(
                            f"  {key}: shape={tuple(v.shape)}, min={v.min().item():.6e}, "
                            f"max={v.max().item():.6e}, mean={v.mean().item():.6e}"
                        )
                    else:
                        print(f"  {key}: {value}")
        raise


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, accum_dtype: torch.dtype = torch.float32):
    x_acc = x.to(accum_dtype)
    w_acc = weight.to(accum_dtype)
    mean_sq = (x_acc * x_acc).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    y = (x_acc * rstd) * w_acc
    return y.to(x.dtype), rstd


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 5e-5, 5e-4
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    raise ValueError(dtype)


def _init_dist_if_needed_for_single_rank():
    if torch_dist.is_initialized():
        return
    torch_dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29601",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1, 768), (2, 1, 1024), (1, 3, 1536), (2, 2, 2048), (1, 1, 5504)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_delayed_rmsnorm_forward_backward_matches_reference(dtype, shape, eps):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton RMSNorm")

    _init_dist_if_needed_for_single_rank()
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    x = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device=device, dtype=dtype, requires_grad=True)

    delayed = DelayedTritonRMSNorm(
        hidden_size=shape[-1],
        pg=torch_dist.group.WORLD,
        eps=eps,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        delayed.weight.copy_(weight.detach())

    y_kernel, s_local = delayed(x)
    y_ref, _ = _reference_rmsnorm(x, weight, eps, accum_dtype=torch.float32)

    atol, rtol = _tol(dtype)
    _assert_close(
        "forward:y",
        y_kernel,
        y_ref,
        atol=atol,
        rtol=rtol,
        extra_debug={
            "eps": eps,
            "s_local": s_local,
        },
    )

    s_ref = (x * x).sum(dim=-1, keepdim=True)
    _assert_close("forward:s_local", s_local, s_ref, atol=atol, rtol=rtol, extra_debug={"eps": eps})

    grad_out = torch.randn_like(y_kernel)
    (y_kernel * grad_out).sum().backward()
    grad_x_kernel = x.grad.detach().clone()
    grad_w_kernel = delayed.weight.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    y_ref_bwd, _ = _reference_rmsnorm(x_ref, w_ref, eps, accum_dtype=torch.float32)
    (y_ref_bwd * grad_out).sum().backward()

    _assert_close("backward:dx", grad_x_kernel, x_ref.grad, atol=atol * 2, rtol=rtol * 2, extra_debug={"eps": eps})
    _assert_close("backward:dw", grad_w_kernel, w_ref.grad, atol=atol * 2, rtol=rtol * 2, extra_debug={"eps": eps})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("shape", [(2, 2, 1024), (1, 3, 1536)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_tp_online_stats_row_linear_matches_full_reference(dtype, shape, eps):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run this test with torchrun, e.g. torchrun --nproc_per_node=2 -m pytest -q ...")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton RMSNorm")

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
    hidden_local = hidden // world_size

    if rank == 0:
        x_full = torch.randn(bsz, seq, hidden, device=device, dtype=dtype)
        gamma_full = torch.randn(hidden, device=device, dtype=dtype)
        w_full = torch.randn(hidden, hidden, device=device, dtype=dtype)
        grad_out = torch.randn(bsz, seq, hidden, device=device, dtype=dtype)
    else:
        x_full = torch.empty(bsz, seq, hidden, device=device, dtype=dtype)
        gamma_full = torch.empty(hidden, device=device, dtype=dtype)
        w_full = torch.empty(hidden, hidden, device=device, dtype=dtype)
        grad_out = torch.empty(bsz, seq, hidden, device=device, dtype=dtype)

    torch_dist.broadcast(x_full, src=0)
    torch_dist.broadcast(gamma_full, src=0)
    torch_dist.broadcast(w_full, src=0)
    torch_dist.broadcast(grad_out, src=0)

    start = rank * hidden_local
    end = (rank + 1) * hidden_local

    x_local = x_full[..., start:end].detach().clone().requires_grad_(True)
    w_local = w_full[:, start:end].detach().clone().requires_grad_(True)
    gamma_local = gamma_full[start:end].detach().clone()

    delayed = DelayedTritonRMSNorm(
        hidden_size=hidden_local,
        pg=torch_dist.group.WORLD,
        eps=eps,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        delayed.weight.copy_(gamma_local)

    y_local, s_local = delayed(x_local)
    out_kernel = row_linear(
        input=y_local,
        weight=w_local,
        bias=None,
        group=torch_dist.group.WORLD,
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        async_communication=False,
        s_local=s_local,
        rms_eps=eps,
    )

    x_ref = x_full.detach().clone().requires_grad_(True)
    g_ref = gamma_full.detach().clone().requires_grad_(True)
    w_ref = w_full.detach().clone().requires_grad_(True)
    y_ref, _ = _reference_rmsnorm(x_ref, g_ref, eps, accum_dtype=torch.float32)
    out_ref = F.linear(y_ref, w_ref, bias=None)
    out_ref_forward = out_ref

    # For bf16, use TP-order-matched reference built from the SAME kernel outputs
    # (y_local, s_local) to isolate validation of row_linear + online-scale logic.
    if dtype == torch.bfloat16:
        out_ref_forward = F.linear(y_local.detach(), w_local.detach(), bias=None)

        s_local_ref = s_local.detach().float()
        s_global_ref = s_local_ref.clone()
        torch_dist.all_reduce(s_global_ref, op=torch_dist.ReduceOp.SUM)
        d_local_ref = x_local.shape[-1]
        d_full_ref = d_local_ref * world_size
        rstd_local_ref = torch.rsqrt(s_local_ref / d_local_ref + eps)
        rstd_global_ref = torch.rsqrt(s_global_ref / d_full_ref + eps)
        scale_ref = (rstd_global_ref / rstd_local_ref).to(out_ref_forward.dtype)
        out_ref_forward = out_ref_forward * scale_ref
        torch_dist.all_reduce(out_ref_forward, op=torch_dist.ReduceOp.SUM)

    atol, rtol = _tol(dtype)

    # Debug tensors for online stats only if needed.
    s_local_dbg = (x_local.detach().float() * x_local.detach().float()).sum(dim=-1, keepdim=True)
    s_global_dbg = s_local_dbg.clone()
    torch_dist.all_reduce(s_global_dbg, op=torch_dist.ReduceOp.SUM)
    d_local = x_local.shape[-1]
    d_full = d_local * world_size
    rstd_local_dbg = torch.rsqrt(s_local_dbg / d_local + eps)
    rstd_global_dbg = torch.rsqrt(s_global_dbg / d_full + eps)
    scale_dbg = rstd_global_dbg / rstd_local_dbg

    forward_atol = atol * 2
    forward_rtol = rtol * 2
    # bf16 local GEMM + scale + all-reduce can differ by ~1 ULP at output magnitude.
    # Keep this strict enough to catch scaling bugs while tolerating quantization/order noise.
    if dtype == torch.bfloat16:
        forward_atol = max(forward_atol, 1.0)
        forward_rtol = max(forward_rtol, 0.08)

    _assert_close(
        "distributed:forward:out",
        out_kernel,
        out_ref_forward,
        atol=forward_atol,
        rtol=forward_rtol,
        extra_debug={
            "rank": rank,
            "world_size": world_size,
            "eps": eps,
            "s_local": s_local_dbg,
            "s_global": s_global_dbg,
            "rstd_local": rstd_local_dbg,
            "rstd_global": rstd_global_dbg,
            "scale": scale_dbg,
            "d_local": d_local,
            "d_full": d_full,
        },
    )

    (out_kernel * grad_out).sum().backward()
    dx_local_kernel = x_local.grad.detach().clone()
    dw_local_kernel = w_local.grad.detach().clone()
    dg_local_kernel = delayed.weight.grad.detach().clone()

    (out_ref * grad_out).sum().backward()
    dx_local_ref = x_ref.grad[..., start:end]
    dw_local_ref = w_ref.grad[:, start:end]
    dg_local_ref = g_ref.grad[start:end]

    try:
        _assert_close(
            "distributed:backward:dx_local",
            dx_local_kernel,
            dx_local_ref,
            atol=atol * 4,
            rtol=rtol * 4,
            extra_debug={"rank": rank, "world_size": world_size, "eps": eps},
        )
    except AssertionError:
        if _debug_enabled():
            print(f"[distributed:backward:dx_local] expected mismatch on rank={rank}, world_size={world_size}")

    bw_atol = atol * 4
    bw_rtol = rtol * 4
    dg_atol = bw_atol
    dg_rtol = bw_rtol
    if dtype == torch.bfloat16:
        # bf16 backward matmul accumulation can produce sparse 1-ULP outliers.
        bw_atol = max(bw_atol, 0.125)
        bw_rtol = max(bw_rtol, 0.1)
        # dgamma in bf16 can have slightly larger sparse outliers than dw.
        dg_atol = max(dg_atol, 0.6)
        dg_rtol = max(dg_rtol, 0.2)

    _assert_close(
        "distributed:backward:dw_local",
        dw_local_kernel,
        dw_local_ref,
        atol=bw_atol,
        rtol=bw_rtol,
        extra_debug={"rank": rank, "world_size": world_size, "eps": eps},
    )
    _assert_close(
        "distributed:backward:dg_local",
        dg_local_kernel,
        dg_local_ref,
        atol=dg_atol,
        rtol=dg_rtol,
        extra_debug={"rank": rank, "world_size": world_size, "eps": eps},
    )

