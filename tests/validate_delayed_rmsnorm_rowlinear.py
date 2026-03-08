"""
Minimal validation script to verify correctness of DelayedRMSNorm + TensorParallelRowLinear
with recovery scale under Tensor Parallelism (TP). Supports iterative validation and training.

Run (forward only, 1 iter):
    torchrun --nproc_per_node=4 nanotron/tests/validate_delayed_rmsnorm_rowlinear.py --mode fwd

Run (forward + backward validation, 10 iters):
    torchrun --nproc_per_node=4 nanotron/tests/validate_delayed_rmsnorm_rowlinear.py --mode both --iters 10

Run (training mode with AdamW, 10 steps):
    torchrun --nproc_per_node=4 nanotron/tests/validate_delayed_rmsnorm_rowlinear.py --train --iters 10 --dtype bf16
"""
import os
import sys
import argparse

import torch
import torch.distributed as dist

from nanotron.nn.layer_norm import TritonRMSNorm, DelayedTritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelRowLinear


def gather_full_from_tp_shards(tensor_local: torch.Tensor, dim: int, tp_size: int, pg: dist.ProcessGroup) -> torch.Tensor:
    """All-gather TP shards and concatenate along dim. Must be called by all TP ranks."""
    tensor_local = tensor_local.contiguous()
    shards = [torch.empty_like(tensor_local) for _ in range(tp_size)]
    shards = [t.contiguous() for t in shards]
    dist.all_gather(shards, tensor_local, group=pg)
    return torch.cat(shards, dim=dim)


def gather_1d_gamma_shards(gamma_local: torch.Tensor, tp_size: int, pg: dist.ProcessGroup) -> torch.Tensor:
    """All-gather 1D gamma shards and concatenate. Must be called by all TP ranks."""
    gamma_local = gamma_local.contiguous()
    shards = [torch.empty_like(gamma_local) for _ in range(tp_size)]
    shards = [t.contiguous() for t in shards]
    dist.all_gather(shards, gamma_local, group=pg)
    return torch.cat(shards, dim=0)


def main():
    # ============================================================================
    # 1. Common Setup
    # ============================================================================
    
    parser = argparse.ArgumentParser(description="Validate DelayedRMSNorm + TP RowLinear (with recovery)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Computation dtype for the test.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fwd", "both"],
        default="both",
        help="Validation mode: forward only, or forward + backward.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Number of iterations to run.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training mode with AdamW optimizer updates.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="In training mode, enforce allclose checks (default: report metrics only).",
    )
    args = parser.parse_args()
    
    # In training mode, force backward
    if args.train:
        args.mode = "both"

    # Initialize distributed
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if world_size != 4:
            print("Warning: Running with world_size != 4. Use torchrun --nproc_per_node=4")
    
    # Create TP process group
    if world_size == 4:
        parallel_context = ParallelContext(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            expert_parallel_size=1,
        )
        tp_pg = parallel_context.tp_pg
        tp_size = 4
    else:
        tp_pg = None
        tp_size = 1
    
    # Fix random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Select dtype
    if args.dtype == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    
    # Define dimensions
    batch = 2
    hidden = 1024
    
    # Create global input on rank 0 (reused across iterations)
    if rank == 0:
        x_global = torch.randn(batch, hidden, device=device, dtype=dtype)
    else:
        x_global = torch.empty(batch, hidden, device=device, dtype=dtype)
    
    # Broadcast to all ranks
    if world_size > 1:
        dist.broadcast(x_global, src=0)
    
    dist.barrier()
    if rank == 0:
        print("=" * 80)
        print("Common Setup Complete")
        print(f"  batch={batch}, hidden={hidden}, tp_size={tp_size}")
        print(f"  Running iters={args.iters}, dtype={args.dtype}, mode={args.mode}, train={args.train}")
        if args.train:
            print(f"  Optimizer: lr={args.lr}, weight_decay={args.weight_decay}, betas=({args.beta1}, {args.beta2}), eps={args.eps}")
        print("=" * 80)
        print()
    
    # ============================================================================
    # 2. Generate Reference Weights (once, reused across iterations)
    # ============================================================================
    
    if rank == 0:
        rmsnorm_ref_init = TritonRMSNorm(hidden_size=hidden, device=device, dtype=dtype)
        rmsnorm_ref_init.weight.data.fill_(1.0)
        rmsnorm_weight_ref = rmsnorm_ref_init.weight.data.clone()
        
        row_linear_ref_init = torch.nn.Linear(
            in_features=hidden,
            out_features=hidden,
            bias=False,
            device=device,
            dtype=dtype,
        )
        torch.nn.init.normal_(row_linear_ref_init.weight, mean=0.0, std=0.02)
        row_linear_weight_ref = row_linear_ref_init.weight.data.clone()
    else:
        rmsnorm_weight_ref = None
        row_linear_weight_ref = None
    
    if world_size > 1:
        if rank != 0:
            rmsnorm_weight_ref = torch.empty(hidden, device=device, dtype=dtype)
            row_linear_weight_ref = torch.empty(hidden, hidden, device=device, dtype=dtype)
        dist.broadcast(rmsnorm_weight_ref, src=0)
        dist.broadcast(row_linear_weight_ref, src=0)
    
    dist.barrier()
    
    # ============================================================================
    # 3. Create Modules and Optimizers (for training mode)
    # ============================================================================
    
    # For training mode, create modules once and reuse across iterations
    # For validation mode, modules are created each iteration
    # Define local dimensions (used in both training and validation)
    in_features_local = hidden // tp_size if world_size == 4 else hidden
    hidden_local = hidden // tp_size if world_size == 4 else hidden
    
    if args.train and world_size == 4:
        # Baseline 1 modules (replicated on all ranks for consistency)
        rmsnorm_ref_train = TritonRMSNorm(hidden_size=hidden, device=device, dtype=dtype)
        rmsnorm_ref_train.weight.data.copy_(rmsnorm_weight_ref)
        
        row_linear_ref_train = torch.nn.Linear(
            in_features=hidden,
            out_features=hidden,
            bias=False,
            device=device,
            dtype=dtype,
        )
        row_linear_ref_train.weight.data.copy_(row_linear_weight_ref)
        
        # Baseline 2 modules
        rmsnorm_tp_train = TritonRMSNorm(hidden_size=hidden, device=device, dtype=dtype)
        rmsnorm_tp_train.weight.data.copy_(rmsnorm_weight_ref)
        
        row_linear_tp_train = TensorParallelRowLinear(
            in_features=hidden,
            out_features=hidden,
            pg=tp_pg,
            mode=TensorParallelLinearMode.ALL_REDUCE,
            bias=False,
            device=device,
            dtype=dtype,
        )
        row_linear_tp_train.weight.data.copy_(
            row_linear_weight_ref[:, rank * in_features_local : (rank + 1) * in_features_local]
        )
        
        # My Method modules
        delayed_rmsnorm_train = DelayedTritonRMSNorm(
            hidden_size=hidden_local,
            pg=tp_pg,
            device=device,
            dtype=dtype,
        )
        delayed_rmsnorm_train.weight.data.copy_(
            rmsnorm_weight_ref[rank * hidden_local : (rank + 1) * hidden_local]
        )
        
        row_linear_delayed_train = TensorParallelRowLinear(
            in_features=hidden,
            out_features=hidden,
            pg=tp_pg,
            mode=TensorParallelLinearMode.ALL_REDUCE,
            bias=False,
            device=device,
            dtype=dtype,
        )
        row_linear_delayed_train.weight.data.copy_(
            row_linear_weight_ref[:, rank * in_features_local : (rank + 1) * in_features_local]
        )
        
        # Create optimizers
        opt_ref = torch.optim.AdamW(
            list(rmsnorm_ref_train.parameters()) + list(row_linear_ref_train.parameters()),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        opt_tp = torch.optim.AdamW(
            list(rmsnorm_tp_train.parameters()) + list(row_linear_tp_train.parameters()),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        opt_del = torch.optim.AdamW(
            list(delayed_rmsnorm_train.parameters()) + list(row_linear_delayed_train.parameters()),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        
        dist.barrier()
    else:
        rmsnorm_ref_train = None
        row_linear_ref_train = None
        rmsnorm_tp_train = None
        row_linear_tp_train = None
        delayed_rmsnorm_train = None
        row_linear_delayed_train = None
        opt_ref = None
        opt_tp = None
        opt_del = None
    
    # ============================================================================
    # 4. Iterative Validation/Training Loop
    # ============================================================================
    
    # Accumulate metrics
    fwd_max_abs_ref_tp_list = []
    fwd_mean_abs_ref_tp_list = []
    fwd_max_abs_ref_delayed_list = []
    fwd_mean_abs_ref_delayed_list = []
    
    bwd_dx_max_abs_tp_list = []
    bwd_dx_mean_abs_tp_list = []
    bwd_dx_max_abs_delayed_list = []
    bwd_dx_mean_abs_delayed_list = []
    bwd_dgamma_max_abs_tp_list = []
    bwd_dgamma_mean_abs_tp_list = []
    bwd_dgamma_max_abs_delayed_list = []
    bwd_dgamma_mean_abs_delayed_list = []
    bwd_dW_max_abs_tp_list = []
    bwd_dW_mean_abs_tp_list = []
    bwd_dW_max_abs_delayed_list = []
    bwd_dW_mean_abs_delayed_list = []
    
    # Training mode metrics
    train_loss_ref_list = []
    train_loss_tp_list = []
    train_loss_delayed_list = []
    train_gamma_max_abs_tp_list = []
    train_gamma_mean_abs_tp_list = []
    train_gamma_max_abs_delayed_list = []
    train_gamma_mean_abs_delayed_list = []
    train_W_max_abs_tp_list = []
    train_W_mean_abs_tp_list = []
    train_W_max_abs_delayed_list = []
    train_W_mean_abs_delayed_list = []
    
    # Set tolerances
    if args.dtype == "fp32":
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 5e-2
        rtol = 5e-2
    
    for iter_idx in range(args.iters):
        if rank == 0:
            print(f"\n--- Iteration {iter_idx + 1}/{args.iters} ---")
        
        # ========================================================================
        # Generate input for this iteration
        # ========================================================================
        
        if args.train:
            # In training mode, generate new input each iteration
            if rank == 0:
                torch.manual_seed(1234 + iter_idx)
                torch.cuda.manual_seed_all(1234 + iter_idx)
                x_global = torch.randn(batch, hidden, device=device, dtype=dtype)
            else:
                x_global = torch.empty(batch, hidden, device=device, dtype=dtype)
            
            if world_size > 1:
                dist.broadcast(x_global, src=0)
            dist.barrier()
        
        # ========================================================================
        # Forward Validation/Training
        # ========================================================================
        
        if args.train and world_size == 4:
            # Training mode: use persistent modules
            # Baseline 1 (run on ALL ranks for consistency)
            x_ref = x_global.clone().detach().requires_grad_(True)
            y_ref = row_linear_ref_train(rmsnorm_ref_train(x_ref))
            
            dist.barrier()
            
            # Baseline 2
            x_tp = x_global.clone().detach().requires_grad_(True)
            x_norm_tp = rmsnorm_tp_train(x_tp)
            x_norm_tp_local = x_norm_tp[:, rank * in_features_local : (rank + 1) * in_features_local].contiguous()
            y_tp_ref = row_linear_tp_train(x_norm_tp_local)
            
            dist.barrier()
            
            # My Method
            x_local = x_global[:, rank * hidden_local : (rank + 1) * hidden_local].clone().detach().requires_grad_(True)
            x_norm_local, s_local = delayed_rmsnorm_train(x_local)
            y_tp_delayed = row_linear_delayed_train(x_norm_local, s_local=s_local)
            
            dist.barrier()
            
            # Forward comparison (training mode also compares outputs)
            if rank == 0 and y_ref is not None and y_tp_ref is not None and y_tp_delayed is not None:
                diff_ref_vs_tp_ref = y_ref.float() - y_tp_ref.float()
                max_abs_ref_tp = torch.max(torch.abs(diff_ref_vs_tp_ref)).item()
                mean_abs_ref_tp = torch.mean(torch.abs(diff_ref_vs_tp_ref)).item()
                
                diff_ref_vs_delayed = y_ref.float() - y_tp_delayed.float()
                max_abs_ref_delayed = torch.max(torch.abs(diff_ref_vs_delayed)).item()
                mean_abs_ref_delayed = torch.mean(torch.abs(diff_ref_vs_delayed)).item()
                
                fwd_max_abs_ref_tp_list.append(max_abs_ref_tp)
                fwd_mean_abs_ref_tp_list.append(mean_abs_ref_tp)
                fwd_max_abs_ref_delayed_list.append(max_abs_ref_delayed)
                fwd_mean_abs_ref_delayed_list.append(mean_abs_ref_delayed)
                
                print(f"  Fwd[{iter_idx+1}]: B1vsB2 max/mean={max_abs_ref_tp:.2e}/{mean_abs_ref_tp:.2e}, "
                      f"B1vsM max/mean={max_abs_ref_delayed:.2e}/{mean_abs_ref_delayed:.2e}")
            
            dist.barrier()
        else:
            # Validation mode: create modules each iteration
            # Baseline 1: Single GPU Reference (TP=1)
            if rank == 0:
                rmsnorm_ref = TritonRMSNorm(hidden_size=hidden, device=device, dtype=dtype)
                rmsnorm_ref.weight.data.copy_(rmsnorm_weight_ref)
                
                row_linear_ref = torch.nn.Linear(
                    in_features=hidden,
                    out_features=hidden,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                row_linear_ref.weight.data.copy_(row_linear_weight_ref)
                
                with torch.no_grad():
                    x_norm_ref = rmsnorm_ref(x_global)
                    y_ref = row_linear_ref(x_norm_ref)
            
            dist.barrier()
        
            # Baseline 2: TP=4, Redundant RMSNorm
            y_tp_ref = None
            if world_size == 4:
                rmsnorm_tp = TritonRMSNorm(hidden_size=hidden, device=device, dtype=dtype)
                rmsnorm_tp.weight.data.copy_(rmsnorm_weight_ref)
                
                row_linear_tp = TensorParallelRowLinear(
                    in_features=hidden,
                    out_features=hidden,
                    pg=tp_pg,
                    mode=TensorParallelLinearMode.ALL_REDUCE,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                row_linear_tp.weight.data.copy_(
                    row_linear_weight_ref[:, rank * in_features_local : (rank + 1) * in_features_local]
                )
                
                dist.barrier()
                
                with torch.no_grad():
                    x_norm_tp = rmsnorm_tp(x_global)
                    x_norm_tp_local = x_norm_tp[:, rank * in_features_local : (rank + 1) * in_features_local].contiguous()
                    y_tp_ref = row_linear_tp(x_norm_tp_local)
            
            dist.barrier()
            
            # My Method: TP=4, DelayedRMSNorm + Recovery Scale
            y_tp_delayed = None
            if world_size == 4:
                delayed_rmsnorm = DelayedTritonRMSNorm(
                    hidden_size=hidden_local,
                    pg=tp_pg,
                    device=device,
                    dtype=dtype,
                )
                delayed_rmsnorm.weight.data.copy_(
                    rmsnorm_weight_ref[rank * hidden_local : (rank + 1) * hidden_local]
                )
                
                row_linear_delayed = TensorParallelRowLinear(
                    in_features=hidden,
                    out_features=hidden,
                    pg=tp_pg,
                    mode=TensorParallelLinearMode.ALL_REDUCE,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                row_linear_delayed.weight.data.copy_(
                    row_linear_weight_ref[:, rank * in_features_local : (rank + 1) * in_features_local]
                )
                
                dist.barrier()
                
                x_local = x_global[:, rank * hidden_local : (rank + 1) * hidden_local].contiguous()
                
                with torch.no_grad():
                    x_norm_local, s_local = delayed_rmsnorm(x_local)
                    y_tp_delayed = row_linear_delayed(x_norm_local, s_local=s_local)
            
            dist.barrier()
        
        # Forward comparison on rank 0 (validation mode only, training mode handles this separately)
        if not args.train and rank == 0 and world_size == 4 and y_tp_ref is not None and y_tp_delayed is not None and y_ref is not None:
            diff_ref_vs_tp_ref = y_ref.float() - y_tp_ref.float()
            max_abs_ref_tp = torch.max(torch.abs(diff_ref_vs_tp_ref)).item()
            mean_abs_ref_tp = torch.mean(torch.abs(diff_ref_vs_tp_ref)).item()
            
            diff_ref_vs_delayed = y_ref.float() - y_tp_delayed.float()
            max_abs_ref_delayed = torch.max(torch.abs(diff_ref_vs_delayed)).item()
            mean_abs_ref_delayed = torch.mean(torch.abs(diff_ref_vs_delayed)).item()
            
            fwd_max_abs_ref_tp_list.append(max_abs_ref_tp)
            fwd_mean_abs_ref_tp_list.append(mean_abs_ref_tp)
            fwd_max_abs_ref_delayed_list.append(max_abs_ref_delayed)
            fwd_mean_abs_ref_delayed_list.append(mean_abs_ref_delayed)
            
            print(f"  Fwd[{iter_idx+1}]: B1vsB2 max/mean={max_abs_ref_tp:.2e}/{mean_abs_ref_tp:.2e}, "
                  f"B1vsM max/mean={max_abs_ref_delayed:.2e}/{mean_abs_ref_delayed:.2e}")
        
        # Ensure all ranks reach this barrier before backward section
        if world_size == 4:
            dist.barrier()
        
        # ========================================================================
        # Backward Validation/Training (if --mode both or --train)
        # ========================================================================
        
        if (args.mode == "both" or args.train) and world_size == 4:
            if args.train:
                # ================================================================
                # Training Mode: Use persistent modules and optimizer steps
                # ================================================================
                
                # Zero gradients
                opt_ref.zero_grad(set_to_none=True)
                opt_tp.zero_grad(set_to_none=True)
                opt_del.zero_grad(set_to_none=True)
                
                # Baseline 1: forward + backward + step (run on ALL ranks for consistency)
                loss_ref = (y_ref.float() ** 2).mean()
                loss_ref.backward()
                opt_ref.step()
                loss_ref_val = loss_ref.item() if rank == 0 else None
                
                dist.barrier()
                
                # Baseline 2: forward + backward + step
                # Note: x_tp, x_norm_tp already created in forward pass
                # Register hook for gradient all-reduce
                def all_reduce_hook(grad):
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_pg)
                    return grad
                x_norm_tp.register_hook(all_reduce_hook)
                
                loss_tp = (y_tp_ref.float() ** 2).mean()
                loss_tp.backward()
                opt_tp.step()
                
                # Synchronize rmsnorm_tp.weight after step (it's replicated)
                dist.broadcast(rmsnorm_tp_train.weight.data, src=0, group=tp_pg)
                
                loss_tp_val = loss_tp.item()
                
                dist.barrier()
                
                # My Method: forward + backward + step
                # Note: x_local already created in forward pass
                loss_delayed = (y_tp_delayed.float() ** 2).mean()
                loss_delayed.backward()
                opt_del.step()
                
                loss_delayed_val = loss_delayed.item()
                
                dist.barrier()
                
                # Gather full parameters (ALL ranks must participate in collectives)
                gamma_ref = rmsnorm_ref_train.weight.data.float()
                gamma_tp = rmsnorm_tp_train.weight.data.float()
                gamma_delayed_full = gather_1d_gamma_shards(
                    delayed_rmsnorm_train.weight.data.float(), tp_size=tp_size, pg=tp_pg
                )
                
                W_ref = row_linear_ref_train.weight.data.float()
                W_tp_full = gather_full_from_tp_shards(
                    row_linear_tp_train.weight.data.float(), dim=1, tp_size=tp_size, pg=tp_pg
                )
                W_delayed_full = gather_full_from_tp_shards(
                    row_linear_delayed_train.weight.data.float(), dim=1, tp_size=tp_size, pg=tp_pg
                )
                
                # Compare parameters and print on rank 0 only
                if rank == 0:
                    # Compare gamma
                    diff_gamma_tp = gamma_ref - gamma_tp
                    diff_gamma_delayed = gamma_ref - gamma_delayed_full
                    train_gamma_max_abs_tp_list.append(diff_gamma_tp.abs().max().item())
                    train_gamma_mean_abs_tp_list.append(diff_gamma_tp.abs().mean().item())
                    train_gamma_max_abs_delayed_list.append(diff_gamma_delayed.abs().max().item())
                    train_gamma_mean_abs_delayed_list.append(diff_gamma_delayed.abs().mean().item())
                    
                    # Compare W
                    diff_W_tp = W_ref - W_tp_full
                    diff_W_delayed = W_ref - W_delayed_full
                    train_W_max_abs_tp_list.append(diff_W_tp.abs().max().item())
                    train_W_mean_abs_tp_list.append(diff_W_tp.abs().mean().item())
                    train_W_max_abs_delayed_list.append(diff_W_delayed.abs().max().item())
                    train_W_mean_abs_delayed_list.append(diff_W_delayed.abs().mean().item())
                    
                    train_loss_ref_list.append(loss_ref_val)
                    train_loss_tp_list.append(loss_tp_val)
                    train_loss_delayed_list.append(loss_delayed_val)
                    
                    print(f"  Train[{iter_idx+1}]: loss={loss_ref_val:.4e}/{loss_tp_val:.4e}/{loss_delayed_val:.4e}, "
                          f"γ B1vsB2/B1vsM max={diff_gamma_tp.abs().max().item():.2e}/{diff_gamma_delayed.abs().max().item():.2e}, "
                          f"W B1vsB2/B1vsM max={diff_W_tp.abs().max().item():.2e}/{diff_W_delayed.abs().max().item():.2e}")
                
                dist.barrier()
            else:
                # ================================================================
                # Validation Mode: Create modules each iteration
                # ================================================================
                
                # Baseline 1 backward (TP=1, reference)
                if rank == 0:
                    rmsnorm_ref_bwd = TritonRMSNorm(hidden_size=hidden, device=device, dtype=dtype)
                    rmsnorm_ref_bwd.weight.data.copy_(rmsnorm_weight_ref)
                    
                    row_linear_ref_bwd = torch.nn.Linear(
                        in_features=hidden,
                        out_features=hidden,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    )
                    row_linear_ref_bwd.weight.data.copy_(row_linear_weight_ref)
                    
                    x_ref = x_global.clone().detach().requires_grad_(True)
                    y_ref_bwd = row_linear_ref_bwd(rmsnorm_ref_bwd(x_ref))
                    loss_ref = (y_ref_bwd.float() ** 2).mean()
                    loss_ref.backward()
                    
                    dx_ref = x_ref.grad.detach().float()
                    dgamma_ref = rmsnorm_ref_bwd.weight.grad.detach().float()
                    dW_ref = row_linear_ref_bwd.weight.grad.detach().float()
                else:
                    dx_ref = None
                    dgamma_ref = None
                    dW_ref = None
                
                dist.barrier()
                
                # Baseline 2 backward (TP=4, redundant RMSNorm)
                rmsnorm_tp_bwd = TritonRMSNorm(hidden_size=hidden, device=device, dtype=dtype)
                rmsnorm_tp_bwd.weight.data.copy_(rmsnorm_weight_ref)
                
                row_linear_tp_bwd = TensorParallelRowLinear(
                    in_features=hidden,
                    out_features=hidden,
                    pg=tp_pg,
                    mode=TensorParallelLinearMode.ALL_REDUCE,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                row_linear_tp_bwd.weight.data.copy_(
                    row_linear_weight_ref[:, rank * in_features_local : (rank + 1) * in_features_local]
                )
                
                x_tp = x_global.clone().detach().requires_grad_(True)
                x_norm_tp_bwd = rmsnorm_tp_bwd(x_tp)
                
                # CRITICAL FIX: Register hook to sum gradients across TP ranks before RMSNorm backward
                def all_reduce_hook(grad):
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_pg)
                    return grad
                x_norm_tp_bwd.register_hook(all_reduce_hook)
                
                x_norm_tp_local_bwd = x_norm_tp_bwd[:, rank * in_features_local : (rank + 1) * in_features_local].contiguous()
                y_tp_bwd = row_linear_tp_bwd(x_norm_tp_local_bwd)
                loss_tp = (y_tp_bwd.float() ** 2).mean()
                loss_tp.backward()
                
                # After hook, dx_tp is full on all ranks
                dx_tp = x_tp.grad.detach().float()
                dgamma_tp = rmsnorm_tp_bwd.weight.grad.detach().float()
                
                dW_tp_local = row_linear_tp_bwd.weight.grad.detach().float()
                dW_tp_full = gather_full_from_tp_shards(dW_tp_local, dim=1, tp_size=tp_size, pg=tp_pg)
                
                dist.barrier()
                
                # My Method backward (TP=4, DelayedRMSNorm + recovery)
                delayed_rmsnorm_bwd = DelayedTritonRMSNorm(
                    hidden_size=hidden_local,
                    pg=tp_pg,
                    device=device,
                    dtype=dtype,
                )
                delayed_rmsnorm_bwd.weight.data.copy_(
                    rmsnorm_weight_ref[rank * hidden_local : (rank + 1) * hidden_local]
                )
                
                row_linear_delayed_bwd = TensorParallelRowLinear(
                    in_features=hidden,
                    out_features=hidden,
                    pg=tp_pg,
                    mode=TensorParallelLinearMode.ALL_REDUCE,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                row_linear_delayed_bwd.weight.data.copy_(
                    row_linear_weight_ref[:, rank * in_features_local : (rank + 1) * in_features_local]
                )
                
                x_local_in = x_global[:, rank * hidden_local : (rank + 1) * hidden_local].clone().detach().requires_grad_(True)
                x_norm_local_bwd, s_local_bwd = delayed_rmsnorm_bwd(x_local_in)
                y_delayed_bwd = row_linear_delayed_bwd(x_norm_local_bwd, s_local=s_local_bwd)
                loss_delayed = (y_delayed_bwd.float() ** 2).mean()
                loss_delayed.backward()
                
                dx_delayed_local = x_local_in.grad.detach().float()
                dx_delayed_full = gather_full_from_tp_shards(dx_delayed_local, dim=-1, tp_size=tp_size, pg=tp_pg)
                
                dgamma_delayed_local = delayed_rmsnorm_bwd.weight.grad.detach().float()
                dgamma_delayed_full = gather_1d_gamma_shards(dgamma_delayed_local, tp_size=tp_size, pg=tp_pg)
                
                dW_delayed_local = row_linear_delayed_bwd.weight.grad.detach().float()
                dW_delayed_full = gather_full_from_tp_shards(dW_delayed_local, dim=1, tp_size=tp_size, pg=tp_pg)
                
                dist.barrier()
                
                # Backward comparison on rank 0
                if rank == 0:
                    # dx
                    diff_dx_tp = dx_ref - dx_tp
                    diff_dx_delayed = dx_ref - dx_delayed_full
                    bwd_dx_max_abs_tp_list.append(diff_dx_tp.abs().max().item())
                    bwd_dx_mean_abs_tp_list.append(diff_dx_tp.abs().mean().item())
                    bwd_dx_max_abs_delayed_list.append(diff_dx_delayed.abs().max().item())
                    bwd_dx_mean_abs_delayed_list.append(diff_dx_delayed.abs().mean().item())
                    
                    # dgamma
                    diff_dgamma_tp = dgamma_ref - dgamma_tp
                    diff_dgamma_delayed = dgamma_ref - dgamma_delayed_full
                    bwd_dgamma_max_abs_tp_list.append(diff_dgamma_tp.abs().max().item())
                    bwd_dgamma_mean_abs_tp_list.append(diff_dgamma_tp.abs().mean().item())
                    bwd_dgamma_max_abs_delayed_list.append(diff_dgamma_delayed.abs().max().item())
                    bwd_dgamma_mean_abs_delayed_list.append(diff_dgamma_delayed.abs().mean().item())
                    
                    # dW
                    diff_dW_tp = dW_ref - dW_tp_full
                    diff_dW_delayed = dW_ref - dW_delayed_full
                    bwd_dW_max_abs_tp_list.append(diff_dW_tp.abs().max().item())
                    bwd_dW_mean_abs_tp_list.append(diff_dW_tp.abs().mean().item())
                    bwd_dW_max_abs_delayed_list.append(diff_dW_delayed.abs().max().item())
                    bwd_dW_mean_abs_delayed_list.append(diff_dW_delayed.abs().mean().item())
                    
                    print(f"  Bwd[{iter_idx+1}]: dx B1vsB2/B1vsM max={diff_dx_tp.abs().max().item():.2e}/{diff_dx_delayed.abs().max().item():.2e}, "
                          f"dγ B1vsB2/B1vsM max={diff_dgamma_tp.abs().max().item():.2e}/{diff_dgamma_delayed.abs().max().item():.2e}, "
                          f"dW B1vsB2/B1vsM max={diff_dW_tp.abs().max().item():.2e}/{diff_dW_delayed.abs().max().item():.2e}")
        
        dist.barrier()
    
    # ============================================================================
    # 5. Final Summary
    # ============================================================================
    
    if rank == 0 and world_size == 4:
        print("\n" + "=" * 80)
        print("Final Summary (across all iterations)")
        print("=" * 80)
        
        if args.train:
            # Training mode summary
            print("\n[Training - Loss]")
            if train_loss_ref_list:
                print(f"  Baseline1: avg={sum(train_loss_ref_list)/len(train_loss_ref_list):.4e}, "
                      f"min={min(train_loss_ref_list):.4e}, max={max(train_loss_ref_list):.4e}")
                print(f"  Baseline2: avg={sum(train_loss_tp_list)/len(train_loss_tp_list):.4e}, "
                      f"min={min(train_loss_tp_list):.4e}, max={max(train_loss_tp_list):.4e}")
                print(f"  MyMethod:  avg={sum(train_loss_delayed_list)/len(train_loss_delayed_list):.4e}, "
                      f"min={min(train_loss_delayed_list):.4e}, max={max(train_loss_delayed_list):.4e}")
            
            print("\n[Training - Parameters: γ (RMSNorm weight)]")
            if train_gamma_max_abs_tp_list:
                print(f"  B1vsB2: max(max_abs)={max(train_gamma_max_abs_tp_list):.2e}, avg(mean_abs)={sum(train_gamma_mean_abs_tp_list)/len(train_gamma_mean_abs_tp_list):.2e}")
                print(f"  B1vsM:  max(max_abs)={max(train_gamma_max_abs_delayed_list):.2e}, avg(mean_abs)={sum(train_gamma_mean_abs_delayed_list)/len(train_gamma_mean_abs_delayed_list):.2e}")
            
            print("\n[Training - Parameters: W (RowLinear weight)]")
            if train_W_max_abs_tp_list:
                print(f"  B1vsB2: max(max_abs)={max(train_W_max_abs_tp_list):.2e}, avg(mean_abs)={sum(train_W_mean_abs_tp_list)/len(train_W_mean_abs_tp_list):.2e}")
                print(f"  B1vsM:  max(max_abs)={max(train_W_max_abs_delayed_list):.2e}, avg(mean_abs)={sum(train_W_mean_abs_delayed_list)/len(train_W_mean_abs_delayed_list):.2e}")
            
            # Optional strict check
            if args.strict:
                train_ok = all(
                    max_abs < atol for max_abs in (
                        train_gamma_max_abs_tp_list + train_gamma_max_abs_delayed_list +
                        train_W_max_abs_tp_list + train_W_max_abs_delayed_list
                    )
                )
                print(f"\n  Strict AllClose (atol={atol}, rtol={rtol}): {train_ok}")
                if not train_ok:
                    print("✗ Training validation FAILED (strict mode)")
                    sys.exit(1)
                else:
                    print("✓ Training validation PASSED (strict mode)")
            else:
                print("\n  (Use --strict to enforce allclose checks)")
        else:
            # Validation mode summary
            # Forward summary
            print("\n[Forward]")
            if fwd_max_abs_ref_tp_list:
                print(f"  B1vsB2: max(max_abs)={max(fwd_max_abs_ref_tp_list):.2e}, avg(mean_abs)={sum(fwd_mean_abs_ref_tp_list)/len(fwd_mean_abs_ref_tp_list):.2e}")
                print(f"  B1vsM:  max(max_abs)={max(fwd_max_abs_ref_delayed_list):.2e}, avg(mean_abs)={sum(fwd_mean_abs_ref_delayed_list)/len(fwd_mean_abs_ref_delayed_list):.2e}")
                
                # Check forward allclose (max_abs should be < atol)
                fwd_ok = all(
                    max_abs < atol for max_abs in fwd_max_abs_ref_tp_list + fwd_max_abs_ref_delayed_list
                )
                print(f"  AllClose (atol={atol}, rtol={rtol}): {fwd_ok}")
            else:
                fwd_ok = True
            
            if args.mode == "both":
                # Backward summary
                print("\n[Backward - dx]")
                if bwd_dx_max_abs_tp_list:
                    print(f"  B1vsB2: max(max_abs)={max(bwd_dx_max_abs_tp_list):.2e}, avg(mean_abs)={sum(bwd_dx_mean_abs_tp_list)/len(bwd_dx_mean_abs_tp_list):.2e}")
                    print(f"  B1vsM:  max(max_abs)={max(bwd_dx_max_abs_delayed_list):.2e}, avg(mean_abs)={sum(bwd_dx_mean_abs_delayed_list)/len(bwd_dx_mean_abs_delayed_list):.2e}")
                
                print("\n[Backward - dγ]")
                if bwd_dgamma_max_abs_tp_list:
                    print(f"  B1vsB2: max(max_abs)={max(bwd_dgamma_max_abs_tp_list):.2e}, avg(mean_abs)={sum(bwd_dgamma_mean_abs_tp_list)/len(bwd_dgamma_mean_abs_tp_list):.2e}")
                    print(f"  B1vsM:  max(max_abs)={max(bwd_dgamma_max_abs_delayed_list):.2e}, avg(mean_abs)={sum(bwd_dgamma_mean_abs_delayed_list)/len(bwd_dgamma_mean_abs_delayed_list):.2e}")
                
                print("\n[Backward - dW]")
                if bwd_dW_max_abs_tp_list:
                    print(f"  B1vsB2: max(max_abs)={max(bwd_dW_max_abs_tp_list):.2e}, avg(mean_abs)={sum(bwd_dW_mean_abs_tp_list)/len(bwd_dW_mean_abs_tp_list):.2e}")
                    print(f"  B1vsM:  max(max_abs)={max(bwd_dW_max_abs_delayed_list):.2e}, avg(mean_abs)={sum(bwd_dW_mean_abs_delayed_list)/len(bwd_dW_mean_abs_delayed_list):.2e}")
                    
                    # Check backward allclose (max_abs should be < atol)
                    bwd_ok = all(
                        max_abs < atol for max_abs in (
                            bwd_dx_max_abs_tp_list + bwd_dx_max_abs_delayed_list +
                            bwd_dgamma_max_abs_tp_list + bwd_dgamma_max_abs_delayed_list +
                            bwd_dW_max_abs_tp_list + bwd_dW_max_abs_delayed_list
                        )
                    )
                    print(f"  AllClose (atol={atol}, rtol={rtol}): {bwd_ok}")
                else:
                    bwd_ok = True
                
                all_ok = fwd_ok and bwd_ok
            else:
                all_ok = fwd_ok
            
            print("\n" + "=" * 80)
            if all_ok:
                print("✓ Validation PASSED")
            else:
                print("✗ Validation FAILED")
                sys.exit(1)
            print("=" * 80)
    
    # Cleanup
    if world_size == 4:
        parallel_context.destroy()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

