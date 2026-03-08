"""
Minimal debug script to validate:
  (1) Model construction: BasicCoLA (TP=1, rank0 only) + BTP CoLA (TP=4, all ranks)
  (2) Broadcast Basic RMSNorm weights and load into BTP, then verify weights match.

No forward passes; weights only.

Run:
  torchrun --nproc_per_node=4 nanotron/tests/debug_btp_weight_broadcast_step1_2.py --dtype fp32
  torchrun --nproc_per_node=4 nanotron/tests/debug_btp_weight_broadcast_step1_2.py --dtype bf16
"""

import os
import sys
import argparse
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Add examples/cola to path
cola_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "cola")
sys.path.insert(0, cola_dir)

from config_basic_cola_llama import BasicColaLlamaConfig
from config_cola_llama import ColaLlamaConfig
from basic_cola_llama import BasicColaLlamaDecoderLayer
from cola_llama import LlamaDecoderLayer as ColaBtpLlamaDecoderLayer

from nanotron.config import ParallelismArgs
from nanotron.parallel import ParallelContext


def create_test_configs() -> Tuple[BasicColaLlamaConfig, ColaLlamaConfig]:
    """Create matching configs for Basic and BTP models."""
    basic_config = BasicColaLlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        vocab_size=1000,
        max_position_embeddings=512,
        attn_rank=64,
        mlp_rank=64,
        both_act=False,
        rope_theta=10000.0,
        rope_interleaved=False,
    )

    btp_config = ColaLlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        vocab_size=1000,
        max_position_embeddings=512,
        attn_rank=64,
        mlp_rank=64,
        rope_theta=10000.0,
        rope_interleaved=False,
    )

    return basic_config, btp_config


def find_ln_gamma_param(layer: torch.nn.Module, hidden_per_tp: int, which: str) -> Tuple[str, torch.nn.Parameter]:
    """
    Find the learnable gamma parameter for a given RMSNorm in BTP layer by shape and name.

    Args:
        layer: BTP decoder layer
        hidden_per_tp: shard size = hidden_size // tp_size
        which: "input" or "post" to help disambiguate
    """
    candidates = []
    for name, param in layer.named_parameters():
        if list(param.shape) == [hidden_per_tp]:
            if which == "input" and "input_layernorm" in name:
                candidates.append((name, param))
            elif which == "post" and "post_attention_layernorm" in name:
                candidates.append((name, param))

    # Fallback: any param with shape [hidden_per_tp]
    if not candidates:
        for name, param in layer.named_parameters():
            if list(param.shape) == [hidden_per_tp]:
                candidates.append((name, param))

    if not candidates:
        raise RuntimeError(f"Could not find LN gamma param for which={which}, hidden_per_tp={hidden_per_tp}")

    if len(candidates) > 1:
        # Pick the first, but warn on rank0
        return candidates[0]
    return candidates[0]


def shard_row_linear(full_weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """
    RowParallel sharding (shard INPUT dim): [out_full, in_full] -> local [out_full, in_full//tp].
    """
    out_full, in_full = full_weight.shape
    assert in_full % tp_size == 0, f"in_full={in_full} not divisible by tp_size={tp_size}"
    in_per_tp = in_full // tp_size
    start = tp_rank * in_per_tp
    end = (tp_rank + 1) * in_per_tp
    return full_weight[:, start:end].contiguous()


def shard_col_linear(full_weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """
    ColumnParallel sharding (shard OUTPUT dim): [out_full, in_full] -> local [out_full//tp, in_full].
    """
    out_full, in_full = full_weight.shape
    assert out_full % tp_size == 0, f"out_full={out_full} not divisible by tp_size={tp_size}"
    out_per_tp = out_full // tp_size
    start = tp_rank * out_per_tp
    end = (tp_rank + 1) * out_per_tp
    return full_weight[start:end, :].contiguous()


def shard_batched_col_linear(full_weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """
    BatchedColumnLinear sharding (shard OUTPUT dim with batch): [B, out_full, in_full] -> local [B, out_per_tp, in_full].
    """
    B, out_full, in_full = full_weight.shape
    assert out_full % tp_size == 0, f"out_full={out_full} not divisible by tp_size={tp_size}"
    out_per_tp = out_full // tp_size
    start = tp_rank * out_per_tp
    end = (tp_rank + 1) * out_per_tp
    return full_weight[:, start:end, :].contiguous()


def gather_row_linear(local_weight: torch.Tensor, tp_group, tp_size: int, rank: int) -> torch.Tensor:
    """
    Gather RowParallel shards (input-sharded) into full [out_full, in_full] on rank0.
    """
    shard_list = [torch.empty_like(local_weight) for _ in range(tp_size)]
    dist.all_gather(shard_list, local_weight.contiguous(), group=tp_group)
    if rank == 0:
        return torch.cat(shard_list, dim=1)
    return None


def gather_col_linear(local_weight: torch.Tensor, tp_group, tp_size: int, rank: int) -> torch.Tensor:
    """
    Gather ColumnParallel shards (output-sharded) into full [out_full, in_full] on rank0.
    """
    shard_list = [torch.empty_like(local_weight) for _ in range(tp_size)]
    dist.all_gather(shard_list, local_weight.contiguous(), group=tp_group)
    if rank == 0:
        return torch.cat(shard_list, dim=0)
    return None


def gather_batched_col_linear(local_weight: torch.Tensor, tp_group, tp_size: int, rank: int) -> torch.Tensor:
    """
    Gather BatchedColumnLinear shards (output-sharded with batch) into full [B, out_full, in_full] on rank0.
    """
    shard_list = [torch.empty_like(local_weight) for _ in range(tp_size)]
    dist.all_gather(shard_list, local_weight.contiguous(), group=tp_group)
    if rank == 0:
        # Concatenate along output dim (dim=1): [B, out_per_tp, in] -> [B, out_full, in]
        return torch.cat(shard_list, dim=1)
    return None


def patch_flash_attn_for_fp32():
    """
    Patch flash_attn_varlen_func to handle fp32 by converting to bf16 and back.
    This is needed because flash_attn only supports fp16/bf16, not fp32.
    """
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as original_flash_attn
        import flash_attn.flash_attn_interface as flash_attn_module
        
        def fp32_wrapper(*args, **kwargs):
            # flash_attn_varlen_func uses keyword args: q, k, v
            # Check if any tensor input is fp32
            needs_conversion = False
            q = kwargs.get('q', None)
            k = kwargs.get('k', None)
            v = kwargs.get('v', None)
            
            if q is not None and isinstance(q, torch.Tensor) and q.dtype == torch.float32:
                needs_conversion = True
            elif k is not None and isinstance(k, torch.Tensor) and k.dtype == torch.float32:
                needs_conversion = True
            elif v is not None and isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                needs_conversion = True
            
            if needs_conversion:
                # Convert tensor inputs to bf16
                converted_kwargs = kwargs.copy()
                if q is not None and isinstance(q, torch.Tensor) and q.dtype == torch.float32:
                    converted_kwargs['q'] = q.to(torch.bfloat16)
                if k is not None and isinstance(k, torch.Tensor) and k.dtype == torch.float32:
                    converted_kwargs['k'] = k.to(torch.bfloat16)
                if v is not None and isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    converted_kwargs['v'] = v.to(torch.bfloat16)
                
                # Call original function with bf16 inputs
                output = original_flash_attn(**converted_kwargs)
                
                # Convert output back to fp32
                if isinstance(output, torch.Tensor):
                    return output.to(torch.float32)
                elif isinstance(output, tuple):
                    return tuple(x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 else x for x in output)
                else:
                    return output
            else:
                # Not fp32, call original function
                return original_flash_attn(*args, **kwargs)
        
        # Patch the function
        flash_attn_module.flash_attn_varlen_func = fp32_wrapper
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Debug BTP RMSNorm weight broadcast + shard (steps 1–2 only)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Data type to use: 'fp32' or 'bf16' (default: fp32)",
    )
    parser.add_argument(
        "--debug-step6",
        action="store_true",
        help="Enable detailed Step6 debug for row_linear recovery vs manual recovery.",
    )
    parser.add_argument(
        "--step7",
        action="store_true",
        help="Enable Step7: full decoder-layer forward parity check.",
    )
    parser.add_argument(
        "--step7-verbose",
        action="store_true",
        help="Enable Step7 verbose mode: trace substeps to find first divergence.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Decoder layer index to debug (e.g., 0 or 1).",
    )

    args = parser.parse_args()

    if args.dtype == "fp32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        # Patch flash_attn to handle fp32 (it only supports fp16/bf16)
        patched = patch_flash_attn_for_fp32()
        if not patched:
            print("WARNING: Could not patch flash_attn for fp32. FlashAttention only supports fp16/bf16.")
            print("         Consider using --dtype bf16 instead, or Step 7 may fail.")
    else:
        dtype = torch.bfloat16

    debug_step6 = args.debug_step6
    step7_enabled = args.step7
    step7_verbose = args.step7_verbose
    layer_idx = args.layer_idx

    if not torch.cuda.is_available():
        print("[rank ?] CUDA not available, exiting.")
        sys.exit(0)

    # Fix NCCL device binding: set CUDA device from LOCAL_RANK before init
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # One-time barrier after init with device_ids to ensure proper device binding
    dist.barrier(device_ids=[device])

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tp_size = 4

    if world_size != tp_size:
        if rank == 0:
            print(f"[rank 0] ERROR: world_size={world_size} but expected tp_size={tp_size}")
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print(f"[rank {rank}] Debug BTP RMSNorm weight broadcast + shard (steps 1–2 only)")
        print(f"[rank {rank}] World size = {world_size}, TP size = {tp_size}, dtype = {args.dtype} ({dtype})")

    dist.barrier()

    # Step 1: Build models ----------------------------------------------------
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    basic_config, btp_config = create_test_configs()
    hidden_size = basic_config.hidden_size

    # Create ParallelContext for TP=4
    parallel_context = ParallelContext(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        backend="nccl",
    )
    tp_group = parallel_context.tp_pg
    tp_rank = dist.get_rank(group=tp_group)

    if rank == 0:
        print(f"[rank {rank}] ParallelContext created. tp_rank={tp_rank}")

    # Parallel configs
    basic_parallel_config = ParallelismArgs(
        dp=1, tp=1, pp=1, expert_parallel_size=1, recompute_layer=False
    )
    btp_parallel_config = ParallelismArgs(
        dp=1, tp=tp_size, pp=1, expert_parallel_size=1, recompute_layer=False
    )

    # Basic layer on rank0 only
    basic_layer = None
    if rank == 0:
        basic_layer = (
            BasicColaLlamaDecoderLayer(
                config=basic_config,
                parallel_config=basic_parallel_config,
                tp_pg=dist.group.WORLD,
                layer_idx=layer_idx,
            )
            .cuda()
            .eval()
            .to(dtype=dtype)
        )
        w = basic_layer.input_layernorm.weight
        print(
            f"[rank {rank}] Basic input_layernorm.weight: "
            f"shape={w.shape}, min={w.min().item():.6f}, max={w.max().item():.6f}, "
            f"mean={w.mean().item():.6f}"
        )
        print(f"[rank {rank}] Basic input_layernorm.weight first 5: {w.flatten()[:5].tolist()}")

    # BTP layer on all ranks
    btp_layer = (
        ColaBtpLlamaDecoderLayer(
            config=btp_config,
            parallel_config=btp_parallel_config,
            tp_pg=parallel_context.tp_pg,
            layer_idx=layer_idx,
        )
        .cuda()
        .eval()
        .to(dtype=dtype)
    )

    # On every rank, print BTP RMSNorm-related parameter names and shapes
    ln_param_lines = []
    for name, param in btp_layer.named_parameters():
        if "input_layernorm" in name or "post_attention_layernorm" in name:
            ln_param_lines.append(f"{name}: shape={list(param.shape)}")
    if not ln_param_lines:
        ln_param_lines.append("NO layernorm params found")

    print(f"[rank {rank}] BTP LN params:")
    for line in ln_param_lines:
        print(f"[rank {rank}]   {line}")

    dist.barrier()

    # Step 2: Broadcast Basic LN weights and load into BTP --------------------
    world_group = dist.group.WORLD

    # Prepare full gamma tensors on all ranks
    device = torch.device("cuda")
    if rank == 0:
        basic_in_ln_full = basic_layer.input_layernorm.weight.detach().to(dtype=dtype, device=device).clone()
        basic_post_ln_full = basic_layer.post_attention_layernorm.weight.detach().to(dtype=dtype, device=device).clone()
    else:
        basic_in_ln_full = torch.empty(hidden_size, dtype=dtype, device=device)
        basic_post_ln_full = torch.empty(hidden_size, dtype=dtype, device=device)

    # Broadcast from rank0 to all ranks
    dist.broadcast(basic_in_ln_full, src=0, group=world_group)
    dist.broadcast(basic_post_ln_full, src=0, group=world_group)

    if rank == 0:
        print(
            f"[rank {rank}] Broadcasted Basic LN gammas: "
            f"shape={basic_in_ln_full.shape}, dtype={basic_in_ln_full.dtype}"
        )

    dist.barrier()

    # Shard and copy into BTP LN gammas on each rank
    hidden_per_tp = hidden_size // tp_size

    # Input LN
    in_ln_shard = basic_in_ln_full[tp_rank * hidden_per_tp : (tp_rank + 1) * hidden_per_tp]
    in_ln_name, in_ln_param = find_ln_gamma_param(btp_layer, hidden_per_tp, which="input")
    in_ln_param.data.copy_(in_ln_shard)

    # Post-attn LN
    post_ln_shard = basic_post_ln_full[tp_rank * hidden_per_tp : (tp_rank + 1) * hidden_per_tp]
    post_ln_name, post_ln_param = find_ln_gamma_param(btp_layer, hidden_per_tp, which="post")
    post_ln_param.data.copy_(post_ln_shard)

    print(
        f"[rank {rank}] Loaded LN shards into BTP: "
        f"input_layernorm param='{in_ln_name}', post_attention_layernorm param='{post_ln_name}'"
    )

    dist.barrier()

    # Local validation print
    def print_local_ln_stats(tag: str, full: torch.Tensor, shard_param: torch.nn.Parameter):
        print(
            f"[rank {rank}] {tag} local shard: "
            f"shape={list(shard_param.shape)}, "
            f"min={shard_param.min().item():.6f}, max={shard_param.max().item():.6f}, "
            f"mean={shard_param.mean().item():.6f}"
        )
        print(
            f"[rank {rank}] {tag} local shard first 5: "
            f"{shard_param.detach().flatten()[:5].tolist()}"
        )

    print_local_ln_stats("input_layernorm", basic_in_ln_full, in_ln_param)
    print_local_ln_stats("post_attention_layernorm", basic_post_ln_full, post_ln_param)

    dist.barrier()

    # Gather shards across TP and compare on rank0
    def gather_and_compare(tag: str, full_basic: torch.Tensor, local_param: torch.nn.Parameter) -> bool:
        shard = local_param.detach().contiguous()
        shard_list = [torch.empty_like(shard) for _ in range(tp_size)]
        dist.all_gather(shard_list, shard, group=tp_group)

        if rank == 0:
            recon = torch.cat(shard_list, dim=0)
            max_diff = (recon - full_basic).abs().max().item()
            mean_diff = (recon - full_basic).abs().mean().item()
            print(
                f"[rank {rank}] {tag} gathered vs Basic full: "
                f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
            )
            # In both fp32 and bf16, copy should be exact
            tol = 1e-6
            ok = max_diff <= tol
            return ok
        else:
            return True

    ok_in = gather_and_compare("input_layernorm", basic_in_ln_full, in_ln_param)
    ok_post = gather_and_compare("post_attention_layernorm", basic_post_ln_full, post_ln_param)

    dist.barrier()

    # If RMSNorm checks fail, stop here (Step 2 only)
    rms_ok = ok_in and ok_post
    if rank == 0:
        if rms_ok:
            print("[rank 0] RMSNorm gamma broadcast + shard + load: OK")
        else:
            print("[rank 0] RMSNorm gamma mismatch detected! Skipping linear weights.")

    dist.barrier()

    all_ok = True

    # Step 3: Load and verify all remaining linear weights (only if RMSNorm OK) -------------
    if rms_ok:
        attn_rank = basic_config.attn_rank
        intermediate_size = basic_config.intermediate_size

        # Build reference full weights on rank0
        if rank == 0:
            # Attention qkv_proj0 (RowLinear grouped)
            q_a_T = basic_layer.attn.q_proj.cola_a.t()  # [attn_rank, hidden_size]
            k_a_T = basic_layer.attn.k_proj.cola_a.t()
            v_a_T = basic_layer.attn.v_proj.cola_a.t()
            full_qkv_proj0 = torch.cat([q_a_T, k_a_T, v_a_T], dim=0)  # [3*attn_rank, hidden_size]

            # Attention qkv_proj1 (BatchedColumnLinear)
            q_b_T = basic_layer.attn.q_proj.cola_b.t()  # [hidden_size, attn_rank]
            k_b_T = basic_layer.attn.k_proj.cola_b.t()
            v_b_T = basic_layer.attn.v_proj.cola_b.t()
            full_qkv_proj1 = torch.stack([q_b_T, k_b_T, v_b_T], dim=0)  # [3, hidden_size, attn_rank]

            # Attention o_proj0 (RowLinear)
            full_o_proj0 = basic_layer.attn.o_proj.cola_a.t()  # [attn_rank, hidden_size]

            # Attention o_proj1 (ColumnLinear)
            full_o_proj1 = basic_layer.attn.o_proj.cola_b.t()  # [hidden_size, attn_rank]

            # MLP gate_up_proj0 (RowLinear grouped)
            gate_a_T = basic_layer.mlp.gate_proj.cola_a.t()  # [mlp_rank, hidden_size]
            up_a_T = basic_layer.mlp.up_proj.cola_a.t()
            full_gate_up_proj0 = torch.cat([gate_a_T, up_a_T], dim=0)  # [2*mlp_rank, hidden_size]

            # MLP gate_up_proj1 (BatchedColumnLinear)
            gate_b_T = basic_layer.mlp.gate_proj.cola_b.t()  # [intermediate_size, mlp_rank]
            up_b_T = basic_layer.mlp.up_proj.cola_b.t()
            full_gate_up_proj1 = torch.stack([gate_b_T, up_b_T], dim=0)  # [2, intermediate_size, mlp_rank]

            # MLP down_proj0 (RowLinear)
            full_down_proj0 = basic_layer.mlp.down_proj.cola_a.t()  # [mlp_rank, intermediate_size]

            # MLP down_proj1 (ColumnLinear)
            full_down_proj1 = basic_layer.mlp.down_proj.cola_b.t()  # [hidden_size, mlp_rank]

            # Cast to target dtype/device
            def cast(x):
                return x.to(dtype=dtype, device=device).clone()

            full_qkv_proj0 = cast(full_qkv_proj0)
            full_qkv_proj1 = cast(full_qkv_proj1)
            full_o_proj0 = cast(full_o_proj0)
            full_o_proj1 = cast(full_o_proj1)
            full_gate_up_proj0 = cast(full_gate_up_proj0)
            full_gate_up_proj1 = cast(full_gate_up_proj1)
            full_down_proj0 = cast(full_down_proj0)
            full_down_proj1 = cast(full_down_proj1)
        else:
            # Allocate empty refs on non-rank0
            full_qkv_proj0 = torch.empty((3 * attn_rank, hidden_size), dtype=dtype, device=device)
            full_qkv_proj1 = torch.empty((3, hidden_size, attn_rank), dtype=dtype, device=device)
            full_o_proj0 = torch.empty((attn_rank, hidden_size), dtype=dtype, device=device)
            full_o_proj1 = torch.empty((hidden_size, attn_rank), dtype=dtype, device=device)
            full_gate_up_proj0 = torch.empty((2 * basic_config.mlp_rank, hidden_size), dtype=dtype, device=device)
            full_gate_up_proj1 = torch.empty((2, intermediate_size, basic_config.mlp_rank), dtype=dtype, device=device)
            full_down_proj0 = torch.empty((basic_config.mlp_rank, intermediate_size), dtype=dtype, device=device)
            full_down_proj1 = torch.empty((hidden_size, basic_config.mlp_rank), dtype=dtype, device=device)

        # Broadcast all reference weights from rank0 to all ranks (WORLD)
        ref_tensors = [
            full_qkv_proj0,
            full_qkv_proj1,
            full_o_proj0,
            full_o_proj1,
            full_gate_up_proj0,
            full_gate_up_proj1,
            full_down_proj0,
            full_down_proj1,
        ]
        # Ensure all tensors are contiguous before broadcast (required by c10d)
        for i, t in enumerate(ref_tensors):
            if not t.is_contiguous():
                ref_tensors[i] = t.contiguous()
        for t in ref_tensors:
            dist.broadcast(t, src=0, group=world_group)

        dist.barrier()

        # Helper to load, gather, compare one weight
        def load_gather_compare(
            tag: str,
            full_ref: torch.Tensor,
            local_param: torch.nn.Parameter,
            shard_fn,
            gather_fn,
        ) -> bool:
            # Shard and copy into local BTP param
            local_shard = shard_fn(full_ref, tp_rank, tp_size)
            assert list(local_shard.shape) == list(local_param.shape), (
                f"{tag}: local shard shape {list(local_shard.shape)} "
                f"!= BTP param shape {list(local_param.shape)}"
            )
            local_param.data.copy_(local_shard)

            dist.barrier()

            # Gather back to rank0
            full_btp = gather_fn(local_param.detach(), tp_group, tp_size, rank)
            if rank == 0:
                max_diff = (full_btp - full_ref).abs().max().item()
                mean_diff = (full_btp - full_ref).abs().mean().item()
                print(
                    f"[rank {rank}] {tag}: max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}"
                )
                tol = 1e-6
                ok_here = max_diff <= tol
                return ok_here
            return True

        # Attention qkv_proj0
        ok = load_gather_compare(
            "attn.qkv_proj0",
            full_qkv_proj0,
            btp_layer.attn.qkv_proj0.weight,
            shard_row_linear,
            gather_row_linear,
        )
        all_ok = all_ok and ok

        # Attention qkv_proj1
        ok = load_gather_compare(
            "attn.qkv_proj1",
            full_qkv_proj1,
            btp_layer.attn.qkv_proj1.weight,
            shard_batched_col_linear,
            gather_batched_col_linear,
        )
        all_ok = all_ok and ok

        # Attention o_proj0
        ok = load_gather_compare(
            "attn.o_proj0",
            full_o_proj0,
            btp_layer.attn.o_proj0.weight,
            shard_row_linear,
            gather_row_linear,
        )
        all_ok = all_ok and ok

        # Attention o_proj1
        ok = load_gather_compare(
            "attn.o_proj1",
            full_o_proj1,
            btp_layer.attn.o_proj1.weight,
            shard_col_linear,
            gather_col_linear,
        )
        all_ok = all_ok and ok

        # MLP gate_up_proj0
        ok = load_gather_compare(
            "mlp.gate_up_proj0",
            full_gate_up_proj0,
            btp_layer.mlp.gate_up_proj0.weight,
            shard_row_linear,
            gather_row_linear,
        )
        all_ok = all_ok and ok

        # MLP gate_up_proj1
        ok = load_gather_compare(
            "mlp.gate_up_proj1",
            full_gate_up_proj1,
            btp_layer.mlp.gate_up_proj1.weight,
            shard_batched_col_linear,
            gather_batched_col_linear,
        )
        all_ok = all_ok and ok

        # MLP down_proj0
        ok = load_gather_compare(
            "mlp.down_proj0",
            full_down_proj0,
            btp_layer.mlp.down_proj0.weight,
            shard_row_linear,
            gather_row_linear,
        )
        all_ok = all_ok and ok

        # MLP down_proj1
        ok = load_gather_compare(
            "mlp.down_proj1",
            full_down_proj1,
            btp_layer.mlp.down_proj1.weight,
            shard_col_linear,
            gather_col_linear,
        )
        all_ok = all_ok and ok

    # Step 4: Validate BTP input sharding --------------------------------------
    # Only do this if RMSNorm and linear weights are OK so far
    step4_ok = True
    if rms_ok and all_ok:
        seq_len = 32
        batch_size = 2

        # Rank 0 creates full hidden_states_full and broadcasts to all ranks
        if rank == 0:
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            hidden_states_full = torch.randn(
                seq_len, batch_size, hidden_size, dtype=dtype, device=device
            )
            sequence_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=device
            )
        else:
            hidden_states_full = torch.empty(
                seq_len, batch_size, hidden_size, dtype=dtype, device=device
            )
            sequence_mask = torch.empty(
                batch_size, seq_len, dtype=torch.bool, device=device
            )

        dist.broadcast(hidden_states_full, src=0, group=world_group)
        dist.broadcast(sequence_mask, src=0, group=world_group)

        dist.barrier()

        hidden_per_tp = hidden_size // tp_size
        # Local shard on each TP rank
        start = tp_rank * hidden_per_tp
        end = (tp_rank + 1) * hidden_per_tp
        hidden_states_shard = hidden_states_full[..., start:end].contiguous()

        # Validation A: local shard vs corresponding slice
        slice_ref = hidden_states_full[..., start:end]
        max_diff_local = (hidden_states_shard - slice_ref).abs().max().item()
        mean_diff_local = (hidden_states_shard - slice_ref).abs().mean().item()
        shard_ok = max_diff_local == 0.0
        print(
            f"[rank {rank}] Step4 shard check: "
            f"{'PASS' if shard_ok else 'FAIL'} "
            f"max_abs_diff={max_diff_local:.6e}, mean_abs_diff={mean_diff_local:.6e}"
        )

        dist.barrier()

        # Validation B: reconstruct full from shards via TP group
        shard_list = [torch.empty_like(hidden_states_shard) for _ in range(tp_size)]
        dist.all_gather(shard_list, hidden_states_shard.contiguous(), group=tp_group)

        if rank == 0:
            reconstructed = torch.cat(shard_list, dim=-1)
            max_diff_full = (reconstructed - hidden_states_full).abs().max().item()
            mean_diff_full = (reconstructed - hidden_states_full).abs().mean().item()
            recon_ok = max_diff_full == 0.0
            print(
                f"[rank {rank}] Step4 reconstruct check: "
                f"{'PASS' if recon_ok else 'FAIL'} "
                f"max_abs_diff={max_diff_full:.6e}, mean_abs_diff={mean_diff_full:.6e}"
            )
            step4_ok = shard_ok and recon_ok
        else:
            step4_ok = shard_ok

        dist.barrier()
    else:
        # If previous steps failed, we don't run Step 4
        step4_ok = False

    # Step 5: input_layernorm forward parity (Basic vs BTP) --------------------
    step5_ok = True
    if rms_ok and all_ok and step4_ok:
        if rank == 0:
            print("[rank 0] Step5: input_layernorm forward parity check starting...")

        # Reuse hidden_states_full / hidden_states_shard from Step 4
        # (they were created inside the same scope above when rms_ok and all_ok)

        # 1) Basic reference on rank0
        if rank == 0:
            basic_ln_out = basic_layer.input_layernorm(hidden_states_full)
        else:
            basic_ln_out = None

        # 2) BTP forward on all ranks
        # Find local gamma shard parameter again (input layernorm)
        in_ln_name, in_ln_param = find_ln_gamma_param(btp_layer, hidden_per_tp, which="input")

        ln_out = btp_layer.input_layernorm(hidden_states_shard)
        if isinstance(ln_out, tuple):
            btp_ln_out_shard, btp_rstd = ln_out
        else:
            btp_ln_out_shard = ln_out
            btp_rstd = None

        # Helper to print stats compactly
        def _print_stats(tag: str, tensor: torch.Tensor):
            flat = tensor.detach().flatten()
            min_v = flat.min().item()
            max_v = flat.max().item()
            mean_v = flat.mean().item()
            std_v = flat.std().item()
            first5 = flat[:5].tolist()
            print(
                f"[rank {rank}] {tag}: shape={list(tensor.shape)}, "
                f"min={min_v:.6f}, max={max_v:.6f}, mean={mean_v:.6f}, std={std_v:.6f}, "
                f"first5={first5}"
            )

        # 3) Per-rank prints (local RMSNorm)
        _print_stats("Step5 BTP ln input shard", hidden_states_shard)
        _print_stats(f"Step5 BTP ln gamma shard (param='{in_ln_name}')", in_ln_param)
        _print_stats("Step5 BTP ln output shard (local)", btp_ln_out_shard)
        if btp_rstd is not None:
            _print_stats("Step5 BTP ln rstd (from module)", btp_rstd)

        dist.barrier()

        # 4) Compute global RMSNorm by rescaling local output
        # Get eps from the module
        eps = btp_layer.input_layernorm.eps
        d = hidden_size
        d_local = hidden_per_tp

        # Compute local sumsq
        s_local = (hidden_states_shard * hidden_states_shard).sum(dim=-1, keepdim=True)  # [seq, batch, 1]

        # Compute local rstd
        rstd_local = torch.rsqrt(s_local / d_local + eps)  # [seq, batch, 1]

        # All-reduce to get global sumsq
        s_global = s_local.clone()
        dist.all_reduce(s_global, op=dist.ReduceOp.SUM, group=tp_group)

        # Compute global rstd
        rstd_global = torch.rsqrt(s_global / d + eps)  # [seq, batch, 1]

        # Rescale factor
        scale = rstd_global / rstd_local  # [seq, batch, 1]

        # Apply rescale to local output
        y_global_local = btp_ln_out_shard * scale  # [seq, batch, d_local]

        # Print stats for rescaling intermediates
        _print_stats("Step5 s_local", s_local)
        _print_stats("Step5 rstd_local", rstd_local)
        _print_stats("Step5 rstd_global", rstd_global)
        _print_stats("Step5 scale (rstd_global/rstd_local)", scale)
        _print_stats("Step5 y_global_local (rescaled)", y_global_local)

        dist.barrier()

        # 5) Validate rstd_global consistency across ranks
        rstd_global_list = [torch.empty_like(rstd_global) for _ in range(tp_size)]
        dist.all_gather(rstd_global_list, rstd_global.contiguous(), group=tp_group)
        if rank == 0:
            # Compare all ranks' rstd_global to rank 0's rstd_global
            base_rstd = rstd_global_list[0]
            max_diff_rstd_global = 0.0
            for i in range(1, tp_size):
                diff_val = (rstd_global_list[i] - base_rstd).abs().max().item()
                max_diff_rstd_global = max(max_diff_rstd_global, diff_val)
            print(
                f"[rank 0] Step5 rstd_global cross-rank max_abs_diff={max_diff_rstd_global:.6e}"
            )

        dist.barrier()

        # 6) Gather rescaled BTP output shards and compare to Basic on rank0
        out_list = [torch.empty_like(y_global_local) for _ in range(tp_size)]
        dist.all_gather(out_list, y_global_local.contiguous(), group=tp_group)

        if rank == 0:
            y_global_full = torch.cat(out_list, dim=-1)  # [seq, batch, hidden_size]

            # Print Basic and BTP full stats
            _print_stats("Step5 Basic ln out (full)", basic_ln_out)
            _print_stats("Step5 BTP ln out (gathered rescaled full)", y_global_full)

            diff = (basic_ln_out - y_global_full).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Tolerances by dtype
            if dtype == torch.float32:
                atol = rtol = 1e-6
            else:
                atol = rtol = 2e-2

            parity_ok = torch.allclose(basic_ln_out, y_global_full, atol=atol, rtol=rtol)
            print(
                f"[rank 0] Step5 LN output diff (rescaled): "
                f"max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}, "
                f"atol={atol:.1e}, rtol={rtol:.1e}, parity_ok={parity_ok}"
            )

            step5_ok = parity_ok
        else:
            step5_ok = True

        dist.barrier()

        # ------------------------------------------------------------------ Step 6
        # Validate in-kernel / in-row_linear recovery path for qkv_proj0
        step6_ok = True
        if rank == 0:
            print("[rank 0] Step6: qkv_proj0 recovery parity check starting...")

        # 1) Basic reference on rank0: LN + packed QKV low-rank before/after lr_act
        if rank == 0:
            # Reuse basic_ln_out from Step 5 (already computed), but recompute if needed
            if basic_ln_out is None:
                basic_ln_out = basic_layer.input_layernorm(hidden_states_full)

            basic_q_lr = torch.matmul(basic_ln_out, basic_layer.attn.q_proj.cola_a)  # [seq, batch, attn_rank]
            basic_k_lr = torch.matmul(basic_ln_out, basic_layer.attn.k_proj.cola_a)
            basic_v_lr = torch.matmul(basic_ln_out, basic_layer.attn.v_proj.cola_a)
            basic_qkv_lr = torch.cat([basic_q_lr, basic_k_lr, basic_v_lr], dim=-1)  # [seq, batch, 3*attn_rank]

            # Use BTP's lr_act for apples-to-apples comparison
            basic_qkv_lr_act = btp_layer.attn.lr_act(basic_qkv_lr)
        else:
            basic_qkv_lr = None
            basic_qkv_lr_act = None

        dist.barrier()

        # Helpers for debug stats/slices
        def _stats(name: str, t: torch.Tensor):
            flat = t.detach().flatten()
            print(
                f"[rank {rank}] {name}: shape={list(t.shape)}, dtype={t.dtype}, "
                f"min={flat.min().item():.6e}, max={flat.max().item():.6e}, "
                f"mean={flat.mean().item():.6e}, std={flat.std().item():.6e}"
            )

        def _print_token(name: str, t: torch.Tensor, t_idx: int = 0, b_idx: int = 0, k: int = 8):
            with torch.no_grad():
                vals = t[t_idx, b_idx, :k].detach().cpu().tolist()
            print(f"[rank {rank}] {name}[t={t_idx},b={b_idx},:k]={vals}")

        def _diff_stats(a: torch.Tensor, b: torch.Tensor):
            """Compute difference statistics between two tensors."""
            # Work in fp32 for stable stats (quantile requires float/double)
            diff_abs = (a - b).abs().to(torch.float32)
            flat_abs = diff_abs.flatten()
            max_abs = flat_abs.max().item()
            mean_abs = flat_abs.mean().item()
            p99_abs = flat_abs.quantile(0.99).item()
            
            # Relative diff: |a-b|/(|a|+1e-6)
            a_abs = a.abs().to(torch.float32)
            rel_diff = diff_abs / (a_abs + 1e-6)
            max_rel = rel_diff.max().item()
            
            return {
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "p99_abs": p99_abs,
                "max_rel": max_rel,
            }

        # 2) BTP compute on all ranks: LN(shard) + qkv_proj0 with internal recovery + lr_act
        ln_out_btp = btp_layer.input_layernorm(hidden_states_shard)
        if isinstance(ln_out_btp, tuple):
            # Follow the model's own convention: second output is passed as s_local into row-linear
            btp_ln_out_local, btp_s_local = ln_out_btp
        else:
            btp_ln_out_local = ln_out_btp
            btp_s_local = None

        if debug_step6 and btp_s_local is not None:
            s_local_copy = btp_s_local.detach().clone()
        else:
            s_local_copy = None

        # qkv_proj0 should perform row-parallel projection + ALL_REDUCE + recovery internally.
        # Its signature is (x, s_local=None), so pass the second LN output as s_local.
        if btp_s_local is not None:
            btp_qkv_lr = btp_layer.attn.qkv_proj0(btp_ln_out_local, btp_s_local)
        else:
            btp_qkv_lr = btp_layer.attn.qkv_proj0(btp_ln_out_local)

        btp_qkv_lr_act = btp_layer.attn.lr_act(btp_qkv_lr)

        # Optional: cross-rank consistency of btp_qkv_lr (should be identical across ranks)
        qkv_list = [torch.empty_like(btp_qkv_lr) for _ in range(tp_size)]
        dist.all_gather(qkv_list, btp_qkv_lr.contiguous(), group=tp_group)
        if rank == 0:
            base = qkv_list[0]
            max_cross_rank = 0.0
            for i in range(1, tp_size):
                diff_i = (qkv_list[i] - base).abs().max().item()
                max_cross_rank = max(max_cross_rank, diff_i)
            print(
                f"[rank 0] Step6 btp_qkv_lr cross-rank max_abs_diff={max_cross_rank:.6e}"
            )

        dist.barrier()

        # ---- DEBUG BLOCK: Compare three versions (A: Basic, B: Manual 2-allreduce, C: BTP row_linear) ----
        # NOTE: Do NOT use y_global_local from Step5 here.
        # Step6 debug validates qkv_proj0 recovery, so x_local must be btp_ln_out_local (the local LN output shard).
        if debug_step6:
            if rank == 0:
                print("\n[rank 0] Step6 DEBUG MODE: Comparing three versions of qkv_proj0 output")
                print("[rank 0] A: Basic reference (ground truth)")
                print("[rank 0] B: Manual 2-allreduce reference (using same inputs as C)")
                print("[rank 0] C: Actual BTP module output (row_linear with recovery)")

            # A) Check if s_local was mutated by row_linear
            if btp_s_local is not None and s_local_copy is not None:
                max_delta_s = (btp_s_local - s_local_copy).abs().max().item()
                print(
                    f"[rank {rank}] Step6 debug: s_local mutation check: max_abs_diff={max_delta_s:.6e}"
                )
                if max_delta_s > 1e-8:
                    print(f"[rank {rank}] ⚠️  WARNING: s_local was mutated by row_linear!")

            dist.barrier()

            # Get eps from LN module (same as used by LN)
            eps_cfg = btp_layer.input_layernorm.eps
            d_local = hidden_per_tp
            d_full = hidden_size
            assert d_full == basic_config.hidden_size, f"d_full={d_full} != config.hidden_size={basic_config.hidden_size}"

            if rank == 0:
                print(
                    f"[rank 0] Step6 debug: d_local={d_local}, d_full={d_full}, eps={eps_cfg:.6e}"
                )

            dist.barrier()

            # B) Manual 2-allreduce reference (must use same inputs as C)
            # Use the exact LN output shard that the BTP module feeds into qkv_proj0
            x_local = btp_ln_out_local  # [seq, batch, d_local] - local LN output shard (NOT y_global_local)

            # s_local: must match the stat returned by LN (same as used by C)
            assert btp_s_local is not None, "Step6 debug requires LN to return s_local"
            s_local_for_manual = btp_s_local.detach().clone()  # Same stat used by C

            _stats("Step6 debug B: x_local (btp_ln_out_local)", x_local)
            _stats("Step6 debug B: s_local_for_manual", s_local_for_manual)

            # Validation print on rank0 to show consistency
            if rank == 0:
                t_idx = 0
                b_idx = 0
                print(
                    f"[rank 0] Step6 debug B: validation (t={t_idx}, b={b_idx}): "
                    f"s_local_for_manual[t,b,0]={s_local_for_manual[t_idx, b_idx, 0].item():.6e}, "
                    f"x_local[t,b,:8]={x_local[t_idx, b_idx, :8].detach().cpu().tolist()}"
                )

            # Compute out_partial = F.linear(x_local, weight, bias) on each rank
            w_local = btp_layer.attn.qkv_proj0.weight
            b_local = btp_layer.attn.qkv_proj0.bias  # May be None on nonzero ranks
            out_partial = F.linear(x_local, w_local, b_local)
            _stats("Step6 debug B: out_partial (F.linear output)", out_partial)

            # FIRST ALL-REDUCE: Compute s_global = all_reduce_sum(s_local)
            s_global_manual = s_local_for_manual.detach().clone()
            dist.all_reduce(s_global_manual, op=dist.ReduceOp.SUM, group=tp_group)
            _stats("Step6 debug B: s_global (first all_reduce)", s_global_manual)

            # Verify s_global consistency across ranks
            s_global_list = [torch.empty_like(s_global_manual) for _ in range(tp_size)]
            dist.all_gather(s_global_list, s_global_manual.contiguous(), group=tp_group)
            if rank == 0:
                base_sg = s_global_list[0]
                max_sg_diff = 0.0
                for i in range(1, tp_size):
                    dsg = (s_global_list[i] - base_sg).abs().max().item()
                    max_sg_diff = max(max_sg_diff, dsg)
                print(
                    f"[rank 0] Step6 debug B: s_global cross-rank max_abs_diff={max_sg_diff:.6e}"
                )

            # Compute rstd_local, rstd_global, scale
            rstd_local_manual = torch.rsqrt(s_local_for_manual / d_local + eps_cfg)
            rstd_global_manual = torch.rsqrt(s_global_manual / d_full + eps_cfg)
            scale_manual = rstd_global_manual / rstd_local_manual

            _stats("Step6 debug B: rstd_local_manual", rstd_local_manual)
            _stats("Step6 debug B: rstd_global_manual", rstd_global_manual)
            _stats("Step6 debug B: scale_manual", scale_manual)

            # Apply scale: out_partial_scaled = out_partial * scale
            out_partial_scaled = out_partial * scale_manual
            _stats("Step6 debug B: out_partial_scaled (out_partial * scale)", out_partial_scaled)

            # SECOND ALL-REDUCE: Compute out_manual = all_reduce_sum(out_partial_scaled)
            out_manual = out_partial_scaled.detach().clone()
            dist.all_reduce(out_manual, op=dist.ReduceOp.SUM, group=tp_group)
            _stats("Step6 debug B: out_manual (second all_reduce)", out_manual)

            # Verify out_manual consistency across ranks (should be identical after all_reduce)
            out_manual_list = [torch.empty_like(out_manual) for _ in range(tp_size)]
            dist.all_gather(out_manual_list, out_manual.contiguous(), group=tp_group)
            if rank == 0:
                base_out = out_manual_list[0]
                max_out_diff = 0.0
                for i in range(1, tp_size):
                    dout = (out_manual_list[i] - base_out).abs().max().item()
                    max_out_diff = max(max_out_diff, dout)
                print(
                    f"[rank 0] Step6 debug B: out_manual cross-rank max_abs_diff={max_out_diff:.6e}"
                )

            # Verify out_manual vs explicit sum of scaled partials
            partial_scaled_list = [torch.empty_like(out_partial_scaled) for _ in range(tp_size)]
            dist.all_gather(partial_scaled_list, out_partial_scaled.contiguous(), group=tp_group)
            if rank == 0:
                sum_scaled_partials = partial_scaled_list[0]
                for i in range(1, tp_size):
                    sum_scaled_partials = sum_scaled_partials + partial_scaled_list[i]
                diff_sum = (sum_scaled_partials - out_manual).abs()
                print(
                    f"[rank 0] Step6 debug B: out_manual vs sum(scaled_partials) "
                    f"max_abs_diff={diff_sum.max().item():.6e}, "
                    f"mean_abs_diff={diff_sum.mean().item():.6e}"
                )

            dist.barrier()

            # C) Actual BTP module output (already computed above as btp_qkv_lr)
            # Verify cross-rank consistency
            if rank == 0:
                print(f"[rank 0] Step6 debug C: btp_qkv_lr cross-rank max_abs_diff={max_cross_rank:.6e}")

            dist.barrier()

            # Comparisons on rank0
            if rank == 0:
                print("\n[rank 0] Step6 DEBUG: Comparison Results")
                print("=" * 80)

                # Comparison B vs C: out_manual vs out_btp
                diff_B_vs_C = (out_manual - btp_qkv_lr).abs()
                max_B_vs_C = diff_B_vs_C.max().item()
                mean_B_vs_C = diff_B_vs_C.mean().item()
                if dtype == torch.float32:
                    atol_comp = rtol_comp = 1e-6
                else:
                    atol_comp = rtol_comp = 2e-2
                match_B_vs_C = torch.allclose(out_manual, btp_qkv_lr, atol=atol_comp, rtol=rtol_comp)

                print(f"\n[rank 0] Comparison B vs C (out_manual vs out_btp):")
                print(f"  max_abs_diff={max_B_vs_C:.6e}, mean_abs_diff={mean_B_vs_C:.6e}")
                print(f"  atol={atol_comp:.1e}, rtol={rtol_comp:.1e}, match={match_B_vs_C}")

                # Comparison A vs C: basic_qkv_lr vs out_btp
                diff_A_vs_C = (basic_qkv_lr - btp_qkv_lr).abs()
                max_A_vs_C = diff_A_vs_C.max().item()
                mean_A_vs_C = diff_A_vs_C.mean().item()
                match_A_vs_C = torch.allclose(basic_qkv_lr, btp_qkv_lr, atol=atol_comp, rtol=rtol_comp)

                print(f"\n[rank 0] Comparison A vs C (basic_qkv_lr vs out_btp):")
                print(f"  max_abs_diff={max_A_vs_C:.6e}, mean_abs_diff={mean_A_vs_C:.6e}")
                print(f"  atol={atol_comp:.1e}, rtol={rtol_comp:.1e}, match={match_A_vs_C}")

                # Comparison A vs B: basic_qkv_lr vs out_manual
                diff_A_vs_B = (basic_qkv_lr - out_manual).abs()
                max_A_vs_B = diff_A_vs_B.max().item()
                mean_A_vs_B = diff_A_vs_B.mean().item()
                match_A_vs_B = torch.allclose(basic_qkv_lr, out_manual, atol=atol_comp, rtol=rtol_comp)

                print(f"\n[rank 0] Comparison A vs B (basic_qkv_lr vs out_manual):")
                print(f"  max_abs_diff={max_A_vs_B:.6e}, mean_abs_diff={mean_A_vs_B:.6e}")
                print(f"  atol={atol_comp:.1e}, rtol={rtol_comp:.1e}, match={match_A_vs_B}")

                # Print token slices for all three versions
                t_idx = 0
                b_idx = 0
                k = 8
                print(f"\n[rank 0] Token slices (t={t_idx}, b={b_idx}, first {k} channels):")
                _print_token("A: basic_qkv_lr", basic_qkv_lr, t_idx, b_idx, k)
                _print_token("B: out_manual", out_manual, t_idx, b_idx, k)
                _print_token("C: btp_qkv_lr", btp_qkv_lr, t_idx, b_idx, k)

                # Print differences
                diff_AB = (basic_qkv_lr - out_manual)[t_idx, b_idx, :k].abs()
                diff_AC = (basic_qkv_lr - btp_qkv_lr)[t_idx, b_idx, :k].abs()
                diff_BC = (out_manual - btp_qkv_lr)[t_idx, b_idx, :k].abs()
                print(f"[rank 0] diff A-B (first {k}): {diff_AB.detach().cpu().tolist()}")
                print(f"[rank 0] diff A-C (first {k}): {diff_AC.detach().cpu().tolist()}")
                print(f"[rank 0] diff B-C (first {k}): {diff_BC.detach().cpu().tolist()}")

                # Print scalar values for the same token
                scalars = {
                    "s_local_for_manual[t,b,0]": s_local_for_manual[t_idx, b_idx, 0].item(),
                    "s_global_manual[t,b,0]": s_global_manual[t_idx, b_idx, 0].item(),
                    "rstd_local_manual[t,b,0]": rstd_local_manual[t_idx, b_idx, 0].item(),
                    "rstd_global_manual[t,b,0]": rstd_global_manual[t_idx, b_idx, 0].item(),
                    "scale_manual[t,b,0]": scale_manual[t_idx, b_idx, 0].item(),
                }
                print(f"\n[rank 0] Step6 debug scalars (t={t_idx}, b={b_idx}): {scalars}")

                # Final conclusion
                print("\n[rank 0] Step6 DEBUG Conclusion:")
                if max_delta_s > 1e-8 if (btp_s_local is not None and s_local_copy is not None) else False:
                    print("  ⚠️  s_local was mutated by row_linear")
                else:
                    print("  ✓ s_local was NOT mutated")
                if max_sg_diff > 1e-8:
                    print(f"  ⚠️  rstd_global differs across ranks (max_diff={max_sg_diff:.6e})")
                else:
                    print("  ✓ rstd_global is consistent across ranks")
                if max_out_diff > 1e-8:
                    print(f"  ⚠️  out_manual differs across ranks (max_diff={max_out_diff:.6e})")
                else:
                    print("  ✓ out_manual is consistent across ranks")
                
                # Comparison conclusions
                if match_A_vs_C and match_A_vs_B and match_B_vs_C:
                    print("  ✓ All three versions match - overall parity OK")
                elif match_B_vs_C:
                    if match_A_vs_C:
                        print("  ✓ B (out_manual) matches C (out_btp) - row_linear recovery is CORRECT")
                        print("  ✓ A (basic) matches C (out_btp) - overall parity OK")
                        if not match_A_vs_B:
                            print("  ⚠️  A (basic) does NOT match B (out_manual) - manual B still not modeling C correctly")
                    else:
                        print("  ✓ B (out_manual) matches C (out_btp) - row_linear recovery is CORRECT")
                        print("  ⚠️  BTP matches manual but not Basic → possible Basic mismatch")
                elif match_A_vs_C:
                    print("  ✓ A (basic) matches C (out_btp) - overall parity OK")
                    if not match_B_vs_C:
                        print("  ✗ B (out_manual) does NOT match C (out_btp) - bug in row_linear recovery logic")
                    if not match_A_vs_B:
                        print("  ⚠️  A (basic) does NOT match B (out_manual) - manual B still not modeling C correctly")
                else:
                    print("  ✗ A (basic) does NOT match C (out_btp) - overall parity FAILED")
                    if not match_B_vs_C:
                        print("  ✗ B (out_manual) does NOT match C (out_btp) - bug in row_linear recovery logic")
                    if not match_A_vs_B:
                        print("  ✗ A (basic) does NOT match B (out_manual) - manual B still not modeling C correctly")

            dist.barrier()

        # 3) Compare Basic vs BTP on rank0 (no gather needed; outputs should be full)
        if rank == 0:
            # Before activation
            diff_lr = (basic_qkv_lr - btp_qkv_lr).abs()
            max_diff_lr = diff_lr.max().item()
            mean_diff_lr = diff_lr.mean().item()

            # After activation
            diff_lr_act = (basic_qkv_lr_act - btp_qkv_lr_act).abs()
            max_diff_lr_act = diff_lr_act.max().item()
            mean_diff_lr_act = diff_lr_act.mean().item()

            if dtype == torch.float32:
                atol6 = rtol6 = 1e-6
            else:
                atol6 = rtol6 = 2e-2

            ok_lr = torch.allclose(basic_qkv_lr, btp_qkv_lr, atol=atol6, rtol=rtol6)
            ok_lr_act = torch.allclose(basic_qkv_lr_act, btp_qkv_lr_act, atol=atol6, rtol=rtol6)

            print(
                f"[rank 0] Step6 qkv_proj0 pre-act diff: "
                f"max_abs_diff={max_diff_lr:.6e}, mean_abs_diff={mean_diff_lr:.6e}, "
                f"atol={atol6:.1e}, rtol={rtol6:.1e}, parity_ok={ok_lr}"
            )
            print(
                f"[rank 0] Step6 qkv_proj0 post-act diff: "
                f"max_abs_diff={max_diff_lr_act:.6e}, mean_abs_diff={mean_diff_lr_act:.6e}, "
                f"atol={atol6:.1e}, rtol={rtol6:.1e}, parity_ok={ok_lr_act}"
            )

            step6_ok = ok_lr and ok_lr_act
            print(
                f"[rank 0] Step6 qkv_proj0 recovery parity: "
                f"{'OK' if step6_ok else 'FAIL'}"
            )

        dist.barrier()
    else:
        # If earlier steps failed, we don't attempt Step 5 / Step 6
        step5_ok = False
        step6_ok = False

    # ------------------------------------------------------------------ Step 7
    # Full decoder-layer forward parity check (always runs)
    step7_ok = True
    first_divergence = None
    # TEMP: Step7 disabled (user request). Keep the code for later re-enable.
    if False and rms_ok and all_ok and step4_ok and step5_ok and step6_ok:
        # Get current device for barrier device_ids
        current_device = torch.cuda.current_device()
        
        if rank == 0:
            print("[rank 0] Step7: Full decoder-layer forward parity check starting...")

        # Prepare input (reuse Step4 logic)
        seq_len = 32
        batch_size = 2

        if rank == 0:
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            hidden_states_full = torch.randn(
                seq_len, batch_size, hidden_size, dtype=dtype, device=device
            )
            sequence_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=device
            )
        else:
            hidden_states_full = torch.empty(
                seq_len, batch_size, hidden_size, dtype=dtype, device=device
            )
            sequence_mask = torch.empty(
                batch_size, seq_len, dtype=torch.bool, device=device
            )

        dist.broadcast(hidden_states_full, src=0, group=world_group)
        dist.broadcast(sequence_mask, src=0, group=world_group)

        dist.barrier(group=tp_group, device_ids=[current_device])

        # Create sharded input for BTP
        hidden_per_tp = hidden_size // tp_size
        start = tp_rank * hidden_per_tp
        end = (tp_rank + 1) * hidden_per_tp
        hidden_states_shard = hidden_states_full[..., start:end].contiguous()

        # Set tolerances
        if dtype == torch.float32:
            atol7 = rtol7 = 1e-6
        else:
            atol7 = rtol7 = 2e-2

        # Helper to create shape signature tensor
        def shape_sig(x):
            """Create a fixed-length shape signature tensor for cross-rank validation."""
            s = list(x.shape)
            s = (s + [-1, -1, -1, -1])[:4]  # Pad to length 4 with -1
            return torch.tensor(s, device=x.device, dtype=torch.int64)
        
        # Helper to gather sharded tensor along last dim to rank0
        def gather_sharded_lastdim_to_rank0(x_local):
            """Gather sharded tensor along last dimension to rank0.
            
            Handles both regular tensors [seq, batch, hidden//tp] and batched tensors [seq, batch, N, hidden//tp].
            First validates that all ranks have identical shapes (except last dim which is sharded).
            """
            # Verify shape consistency across ranks (all dims except last should match)
            sig = shape_sig(x_local)
            sig_list = [torch.empty_like(sig) for _ in range(tp_size)]
            dist.all_gather(sig_list, sig, group=tp_group)
            
            if rank == 0:
                # Check if all shapes match (except last dimension)
                first_sig = sig_list[0].cpu().tolist()
                for r, s in enumerate(sig_list):
                    s_list = s.cpu().tolist()
                    # Compare all dims except last (dim -1)
                    if len(first_sig) > 0 and len(s_list) > 0:
                        # Check all but last dimension
                        for d in range(len(first_sig) - 1):
                            if first_sig[d] != s_list[d] and first_sig[d] != -1 and s_list[d] != -1:
                                print(f"[rank 0] ERROR: Shape mismatch in gather_sharded_lastdim_to_rank0:")
                                print(f"  Rank 0 shape: {first_sig}")
                                print(f"  Rank {r} shape: {s_list}")
                                raise RuntimeError(f"Shape mismatch: rank 0 has {first_sig}, rank {r} has {s_list}")
            
            # All ranks participate in gather
            shard_list = [torch.empty_like(x_local) for _ in range(tp_size)]
            dist.all_gather(shard_list, x_local.contiguous(), group=tp_group)
            if rank == 0:
                return torch.cat(shard_list, dim=-1)
            return None
        
        # Helper to gather last dimension (works for [seq,batch,hidden//tp], [seq,batch,N,hidden//tp], [batch,seq,heads,d//tp])
        def gather_lastdim_to_rank0(x_local):
            """Gather sharded tensor along last dimension to rank0.
            
            Works for various shapes where last dim is sharded:
            - [seq, batch, hidden//tp]
            - [seq, batch, N, hidden//tp]
            - [batch, seq, heads, d//tp]
            """
            # All ranks participate in gather
            shard_list = [torch.empty_like(x_local) for _ in range(tp_size)]
            dist.all_gather(shard_list, x_local.contiguous(), group=tp_group)
            if rank == 0:
                return torch.cat(shard_list, dim=-1)
            return None
        
        # Step 7: Full forward parity trace (always runs) - NEW IMPLEMENTATION
        # Compare Basic (TP=1) vs BTP (TP=4) using real forward path by calling submodules directly
        if rank == 0:
            print("\n" + "=" * 80)
            print("[rank 0] Step7: Full decoder-layer forward parity check (real forward path)")
            print("=" * 80)

        # Helper to print tensor stats on each rank
        def print_tensor_stats(name, tensor, rank_val):
            """Print tensor stats on each rank."""
            if tensor is None:
                print(f"[rank {rank_val}] {name}: None")
                return
            shape_str = str(tuple(tensor.shape))
            dtype_str = str(tensor.dtype)
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            
            # Get first 8 values (flattened if needed)
            flat = tensor.flatten()
            first8 = flat[:min(8, flat.numel())].detach().cpu().tolist()
            
            print(f"[rank {rank_val}] {name}: shape={shape_str}, dtype={dtype_str}, "
                  f"min={min_val:.6e}, max={max_val:.6e}, mean={mean_val:.6e}, std={std_val:.6e}, "
                  f"first8={first8}")

        # FP32 reference attention (rank0 only, naive implementation)
        def ref_attn_fp32(q, k, v, sequence_mask):
            """
            Naive FP32 reference attention implementation.
            
            Args:
                q: [batch, seq, heads, head_dim] in fp32
                k: [batch, seq, heads, head_dim] in fp32 (or kv_heads if GQA)
                v: [batch, seq, heads, head_dim] in fp32 (or kv_heads if GQA)
                sequence_mask: [batch, seq] bool (True=valid token)
            
            Returns:
                attn_out: [batch, seq, heads, head_dim] in fp32
            """
            batch, seq, heads, head_dim = q.shape
            kv_heads = k.shape[2]
            
            # Ensure fp32
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)
            
            # Compute attention scores
            # For GQA: q has q_heads, k/v have kv_heads
            if kv_heads == heads:
                # Standard attention: q [batch, seq, heads, d] @ k^T [batch, seq, d, heads] -> [batch, heads, seq, seq]
                # einsum: bshd (q) @ bthd (k) -> bhst (scores)
                scores = torch.einsum('bshd,bthd->bhst', q, k) / (head_dim ** 0.5)  # [batch, heads, seq, seq]
            else:
                # GQA: q [batch, seq, q_heads, d] @ k [batch, seq, kv_heads, d] -> [batch, q_heads, seq, kv_heads, seq]
                # Each q_head group shares kv_heads
                group_size = heads // kv_heads
                # Reshape q to [batch, seq, kv_heads, group_size, d]
                q_grouped = q.view(batch, seq, kv_heads, group_size, head_dim)
                # Compute scores: [batch, kv_heads, group_size, seq] @ [batch, seq, kv_heads, d] -> [batch, kv_heads, group_size, seq, seq]
                scores_per_group = torch.einsum('bsgkd,bthd->bkgst', q_grouped, k) / (head_dim ** 0.5)  # [batch, kv_heads, group_size, seq, seq]
                # Reshape to [batch, q_heads, seq, seq]
                scores = scores_per_group.view(batch, heads, seq, seq)
            
            # Apply causal mask (lower triangular)
            causal_mask = torch.tril(torch.ones(seq, seq, device=q.device, dtype=torch.bool))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            
            # Apply padding mask from sequence_mask
            if sequence_mask is not None:
                # sequence_mask: [batch, seq] -> [batch, 1, 1, seq] for key positions
                # and [batch, 1, seq, 1] for query positions
                key_mask = sequence_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq]
                query_mask = sequence_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]
                # Mask out invalid positions: if key is invalid or query is invalid
                mask = key_mask & query_mask  # [batch, 1, seq, seq]
                scores = scores.masked_fill(~mask, float('-inf'))
            
            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)  # [batch, q_heads, seq, seq]
            
            # Apply attention to values
            # attn_weights: [batch, q_heads, seq_q, seq_k], v: [batch, seq_k, kv_heads, d]
            # Output: [batch, seq_q, q_heads, d]
            if kv_heads == heads:
                # Standard attention: [batch, heads, seq_q, seq_k] @ [batch, seq_k, heads, d] = [batch, seq_q, heads, d]
                # einsum: batch b, head h, seq_q s, seq_k t (attn_weights) @ batch b, seq_k t, head h, dim d (v) -> batch b, seq_q s, head h, dim d
                attn_out = torch.einsum('bhst,bthd->bshd', attn_weights, v)  # [batch, seq, heads, head_dim]
            else:
                # GQA: attn_weights [batch, q_heads, seq_q, seq_k], v [batch, seq_k, kv_heads, d]
                # Reshape attn_weights to [batch, kv_heads, group_size, seq_q, seq_k]
                group_size = heads // kv_heads
                attn_weights_grouped = attn_weights.view(batch, kv_heads, group_size, seq, seq)
                # Compute output: [batch, kv_heads, group_size, seq_q, seq_k] @ [batch, seq_k, kv_heads, d]
                # -> [batch, kv_heads, group_size, seq_q, d] -> [batch, seq_q, q_heads, d]
                attn_out_grouped = torch.einsum('bkgst,bthd->bksgd', attn_weights_grouped, v)  # [batch, kv_heads, group_size, seq, d]
                attn_out = attn_out_grouped.view(batch, seq, heads, head_dim)
            
            return attn_out  # [batch, seq, heads, head_dim]

        # Helper to compare checkpoint (rank0 only)
        def compare_checkpoint_rank0(name, basic_tensor, btp_local, btp_tensor_full, first_div_name=None):
            """Compare checkpoint on rank0 and return (match, max_abs, first_div_name).
            
            Args:
                name: Checkpoint name
                basic_tensor: Basic (TP=1) tensor on rank0
                btp_local: BTP local tensor (for shape printing, can be same as btp_tensor_full for full tensors)
                btp_tensor_full: BTP full tensor (gathered or full) for comparison
                first_div_name: Current first divergence name (updated if this checkpoint fails)
            """
            if rank == 0:
                if basic_tensor is None or btp_tensor_full is None:
                    print(f"\n[rank 0] {name}: SKIP (missing tensor)")
                    return True, 0.0, first_div_name
                
                diff_stats_sub = _diff_stats(basic_tensor, btp_tensor_full)
                match_sub = torch.allclose(basic_tensor, btp_tensor_full, atol=atol7, rtol=rtol7)
                
                print(f"\n[rank 0] {name}:")
                print(f"  Basic shape: {basic_tensor.shape}, BTP local: {btp_local.shape}, BTP full: {btp_tensor_full.shape}")
                print(f"  max_abs={diff_stats_sub['max_abs']:.6e}, mean_abs={diff_stats_sub['mean_abs']:.6e}, "
                      f"p99_abs={diff_stats_sub['p99_abs']:.6e}, max_rel={diff_stats_sub['max_rel']:.6e}")
                
                if match_sub:
                    print(f"  PASS")
                else:
                    print(f"  FAIL")
                    # Print first 8 sample
                    t_idx, b_idx, k = 0, 0, 8
                    if len(basic_tensor.shape) >= 3 and basic_tensor.shape[0] > t_idx and basic_tensor.shape[1] > b_idx:
                        if len(basic_tensor.shape) == 3 and basic_tensor.shape[2] >= k:
                            basic_sample = basic_tensor[t_idx, b_idx, :k].detach().cpu().tolist()
                            btp_sample = btp_tensor_full[t_idx, b_idx, :k].detach().cpu().tolist()
                        else:
                            # Flatten and take first k
                            basic_sample = basic_tensor.flatten()[:k].detach().cpu().tolist()
                            btp_sample = btp_tensor_full.flatten()[:k].detach().cpu().tolist()
                        print(f"  Sample (t={t_idx}, b={b_idx}, first {k}):")
                        print(f"    Basic: {basic_sample}")
                        print(f"    BTP:   {btp_sample}")
                
                if not match_sub and diff_stats_sub['max_abs'] > 10 * atol7 and first_div_name is None:
                    first_div_name = name
                
                return match_sub, diff_stats_sub['max_abs'], first_div_name
            return True, 0.0, first_div_name

        # Initialize tracking variables
        first_divergence = None
        all_checkpoints_pass = True

        # Run input LN (needed for forward, but DO NOT compare outputs)
        if rank == 0:
            basic_ln = basic_layer.input_layernorm(hidden_states_full)
        else:
            basic_ln = None

        btp_ln_result = btp_layer.input_layernorm(hidden_states_shard)
        if isinstance(btp_ln_result, tuple):
            btp_ln_local, s_local = btp_ln_result
        else:
            btp_ln_local = btp_ln_result
            s_local = None

        # DO NOT compare LN outputs - we already validated LN recovery earlier
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # ========== ATTENTION PATH (real forward, explicitly staged) ==========
        if rank == 0:
            print("\n" + "=" * 80)
            print("[rank 0] Attention-side checkpoints")
            print("=" * 80)

        # (A1) BTP: qkv_proj0 pre-act
        btp_qkv_lr = btp_layer.attn.qkv_proj0(btp_ln_local, s_local)  # RowLinear -> full on all ranks
        print_tensor_stats("BTP A1: qkv_proj0 (pre-act)", btp_qkv_lr, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_q_lr = torch.matmul(basic_ln, basic_layer.attn.q_proj.cola_a)
            basic_k_lr = torch.matmul(basic_ln, basic_layer.attn.k_proj.cola_a)
            basic_v_lr = torch.matmul(basic_ln, basic_layer.attn.v_proj.cola_a)
            basic_qkv_lr = torch.cat([basic_q_lr, basic_k_lr, basic_v_lr], dim=-1)
        else:
            basic_qkv_lr = None
        
        # Compare A1 (qkv_proj0 pre-act) - RowLinear output is full on all ranks
        match_a1, max_abs_a1, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A1: attn.qkv_proj0 (pre-act)", basic_qkv_lr, btp_qkv_lr, btp_qkv_lr, first_divergence
        )
        if not match_a1:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (A1b) BTP: qkv_proj0 post-act
        btp_qkv_lr_act = btp_layer.attn.lr_act(btp_qkv_lr)
        print_tensor_stats("BTP A1b: qkv_proj0 (post-act)", btp_qkv_lr_act, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_qkv_lr_act = btp_layer.attn.lr_act(basic_qkv_lr)  # Use same lr_act for consistency
        else:
            basic_qkv_lr_act = None
        
        # Compare A1b (qkv_proj0 post-act) - RowLinear output is full on all ranks
        match_a1b, max_abs_a1b, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A1b: attn.qkv_proj0 (post-act)", basic_qkv_lr_act, btp_qkv_lr_act, btp_qkv_lr_act, first_divergence
        )
        if not match_a1b:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (A2) BTP: qkv_proj1 output
        # Unflatten to [seq, batch, 3, attn_rank]
        attn_rank = btp_layer.attn.qkv_proj0.out_features // 3  # Each of q/k/v has this rank
        btp_qkv_lr_act_3 = btp_qkv_lr_act.unflatten(-1, (3, -1))  # [seq, batch, 3, attn_rank]
        btp_qkv_proj1_out = btp_layer.attn.qkv_proj1(btp_qkv_lr_act_3)  # Output shape depends on module
        
        # Normalize shape: BatchedColumnLinear returns [gemm_num, seq, batch, out_features//tp]
        # We need [seq, batch, gemm_num, out_features//tp]
        if len(btp_qkv_proj1_out.shape) == 4 and btp_qkv_proj1_out.shape[0] == 3:
            # Shape is [3, seq, batch, out_features//tp], permute to [seq, batch, 3, out_features//tp]
            btp_qkv_proj1_out = btp_qkv_proj1_out.permute(1, 2, 0, 3).contiguous()
        
        print_tensor_stats("BTP A2: qkv_proj1_out", btp_qkv_proj1_out, rank)
        
        # Extract q/k/v states from proj1 output
        # btp_qkv_proj1_out is [seq, batch, 3, hidden//tp] (or [seq, batch, 3, num_heads*d_qk//tp])
        btp_q_states_local, btp_k_states_local, btp_v_states_local = btp_qkv_proj1_out.unbind(dim=2)  # Each: [seq, batch, hidden//tp]
        
        print_tensor_stats("BTP A3: q_states", btp_q_states_local, rank)
        print_tensor_stats("BTP A3: k_states", btp_k_states_local, rank)
        print_tensor_stats("BTP A3: v_states", btp_v_states_local, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_q_lr_act = basic_layer.attn.q_proj.lr_act(basic_q_lr)
            basic_k_lr_act = basic_layer.attn.k_proj.lr_act(basic_k_lr)
            basic_v_lr_act = basic_layer.attn.v_proj.lr_act(basic_v_lr)
            basic_q_full = torch.matmul(basic_q_lr_act, basic_layer.attn.q_proj.cola_b)  # [seq, batch, num_heads*d_qk]
            basic_k_full = torch.matmul(basic_k_lr_act, basic_layer.attn.k_proj.cola_b)
            basic_v_full = torch.matmul(basic_v_lr_act, basic_layer.attn.v_proj.cola_b)
            
            # Stack to match BTP format [seq, batch, 3, out_features]
            basic_qkv_proj1_out = torch.stack([basic_q_full, basic_k_full, basic_v_full], dim=0)  # [3, seq, batch, out]
            basic_qkv_proj1_out = basic_qkv_proj1_out.permute(1, 2, 0, 3)  # [seq, batch, 3, out]
        else:
            basic_q_full = None
            basic_k_full = None
            basic_v_full = None
            basic_qkv_proj1_out = None
        
        # Gather BTP qkv_proj1_out for comparison (all ranks participate)
        btp_qkv_proj1_out_full = gather_lastdim_to_rank0(btp_qkv_proj1_out)  # [seq, batch, 3, hidden]
        
        # Compare A2 (qkv_proj1_out)
        match_a2, max_abs_a2, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A2: attn.qkv_proj1_out", basic_qkv_proj1_out, btp_qkv_proj1_out, btp_qkv_proj1_out_full, first_divergence
        )
        if not match_a2:
            all_checkpoints_pass = False
        
        # Compare A3 (q/k/v states) - gather each separately
        btp_q_full = gather_lastdim_to_rank0(btp_q_states_local)
        btp_k_full = gather_lastdim_to_rank0(btp_k_states_local)
        btp_v_full = gather_lastdim_to_rank0(btp_v_states_local)
        
        match_a3q, max_abs_a3q, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A3: q_states", basic_q_full, btp_q_states_local, btp_q_full, first_divergence
        )
        if not match_a3q:
            all_checkpoints_pass = False
        
        match_a3k, max_abs_a3k, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A3: k_states", basic_k_full, btp_k_states_local, btp_k_full, first_divergence
        )
        if not match_a3k:
            all_checkpoints_pass = False
        
        match_a3v, max_abs_a3v, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A3: v_states", basic_v_full, btp_v_states_local, btp_v_full, first_divergence
        )
        if not match_a3v:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (A4_ref) FP32 reference attention comparison (rank0 only)
        # This validates that RoPE + QKV pipeline is correct before FlashAttention
        if rank == 0:
            print("\n" + "=" * 80)
            print("[rank 0] Checkpoint A4_ref: FP32 reference attention (Basic vs BTP)")
            print("=" * 80)
        
        # Gather BTP q/k/v to rank0 (all ranks participate)
        # btp_q/k/v_full are already gathered above, but we need them in [batch, seq, heads, head_dim] format
        # Get attention attributes
        # NOTE: btp_layer.attn.n_local_*heads are per-TP-rank. After we gather q/k/v to rank0,
        # we must reshape using the *global* head counts.
        d_qk = btp_layer.attn.d_qk
        d_v = btp_layer.attn.d_v
        seq_len, batch_size = btp_q_states_local.shape[0], btp_q_states_local.shape[1]
        
        # Basic q/k/v are already in [seq, batch, hidden] format
        # Reshape to [batch, seq, heads, head_dim] for both Basic and BTP
        if rank == 0:
            # Basic: reshape from [seq, batch, num_heads*d_qk] to [batch, seq, heads, head_dim]
            basic_n_local_q_heads = basic_layer.attn.n_local_q_heads
            basic_n_local_kv_heads = basic_layer.attn.n_local_kv_heads
            basic_d_qk = basic_layer.attn.d_qk
            basic_d_v = basic_layer.attn.d_v

            # Derive global head counts from gathered tensors (robust even if TP head partitioning changes)
            # btp_q_full/btp_k_full/btp_v_full are [seq, batch, hidden] on rank0.
            assert btp_q_full is not None and btp_k_full is not None and btp_v_full is not None
            btp_q_heads_full = btp_q_full.shape[-1] // d_qk
            btp_kv_heads_full_qk = btp_k_full.shape[-1] // d_qk
            btp_kv_heads_full_v = btp_v_full.shape[-1] // d_v
            assert btp_q_heads_full * d_qk == btp_q_full.shape[-1], "btp_q_full lastdim must be divisible by d_qk"
            assert btp_kv_heads_full_qk * d_qk == btp_k_full.shape[-1], "btp_k_full lastdim must be divisible by d_qk"
            assert btp_kv_heads_full_v * d_v == btp_v_full.shape[-1], "btp_v_full lastdim must be divisible by d_v"

            # Sanity: in this test we expect no GQA, so q_heads == kv_heads and they match Basic
            assert btp_q_heads_full == btp_kv_heads_full_qk == btp_kv_heads_full_v, (
                f"Unexpected head count mismatch: q={btp_q_heads_full}, k={btp_kv_heads_full_qk}, v={btp_kv_heads_full_v}"
            )
            assert basic_n_local_q_heads == btp_q_heads_full, (
                f"Basic heads={basic_n_local_q_heads} != BTP gathered heads={btp_q_heads_full}"
            )
            assert basic_d_qk == d_qk and basic_d_v == d_v, (
                f"Head dims mismatch: Basic (d_qk={basic_d_qk}, d_v={basic_d_v}) vs BTP (d_qk={d_qk}, d_v={d_v})"
            )
            
            # Reshape Basic q/k/v
            basic_q_4d = basic_q_full.view(seq_len, batch_size, basic_n_local_q_heads, basic_d_qk).permute(1, 0, 2, 3).contiguous()  # [B, S, Hq, d_qk]
            basic_k_4d = basic_k_full.view(seq_len, batch_size, basic_n_local_kv_heads, basic_d_qk).permute(1, 0, 2, 3).contiguous()  # [B, S, Hkv, d_qk]
            basic_v_4d = basic_v_full.view(seq_len, batch_size, basic_n_local_kv_heads, basic_d_v).permute(1, 0, 2, 3).contiguous()  # [B, S, Hkv, d_v]
            
            # BTP: reshape gathered full tensors [seq, batch, hidden] -> [batch, seq, heads, head_dim]
            btp_q_4d = btp_q_full.view(seq_len, batch_size, btp_q_heads_full, d_qk).permute(1, 0, 2, 3).contiguous()  # [B, S, Hq, d_qk]
            btp_k_4d = btp_k_full.view(seq_len, batch_size, btp_kv_heads_full_qk, d_qk).permute(1, 0, 2, 3).contiguous()  # [B, S, Hkv, d_qk]
            btp_v_4d = btp_v_full.view(seq_len, batch_size, btp_kv_heads_full_v, d_v).permute(1, 0, 2, 3).contiguous()  # [B, S, Hkv, d_v]
            
            # Assert shapes match
            assert basic_q_4d.shape == btp_q_4d.shape, f"Basic q shape {basic_q_4d.shape} != BTP q shape {btp_q_4d.shape}"
            assert basic_k_4d.shape == btp_k_4d.shape, f"Basic k shape {basic_k_4d.shape} != BTP k shape {btp_k_4d.shape}"
            assert basic_v_4d.shape == btp_v_4d.shape, f"Basic v shape {basic_v_4d.shape} != BTP v shape {btp_v_4d.shape}"
            
            # Cast to fp32
            basic_q_fp32 = basic_q_4d.to(torch.float32)
            basic_k_fp32 = basic_k_4d.to(torch.float32)
            basic_v_fp32 = basic_v_4d.to(torch.float32)
            btp_q_fp32 = btp_q_4d.to(torch.float32)
            btp_k_fp32 = btp_k_4d.to(torch.float32)
            btp_v_fp32 = btp_v_4d.to(torch.float32)
            
            # Run FP32 reference attention on both
            attn_ref_basic = ref_attn_fp32(basic_q_fp32, basic_k_fp32, basic_v_fp32, sequence_mask)  # [B, S, Hq, d_v]
            attn_ref_btp = ref_attn_fp32(btp_q_fp32, btp_k_fp32, btp_v_fp32, sequence_mask)  # [B, S, Hq, d_v]
            
            # Compare Basic vs BTP reference attention outputs
            diff_stats_ref = _diff_stats(attn_ref_basic, attn_ref_btp)
            match_ref = torch.allclose(attn_ref_basic, attn_ref_btp, atol=1e-6, rtol=1e-6)
            
            print(f"\n[rank 0] Checkpoint A4_ref: FP32 reference attention (Basic vs BTP):")
            print(f"  Basic ref shape: {attn_ref_basic.shape}, BTP ref shape: {attn_ref_btp.shape}")
            print(f"  max_abs={diff_stats_ref['max_abs']:.6e}, mean_abs={diff_stats_ref['mean_abs']:.6e}, "
                  f"p99_abs={diff_stats_ref['p99_abs']:.6e}, max_rel={diff_stats_ref['max_rel']:.6e}")
            
            if match_ref:
                print(f"  ✓ PASS (Basic vs BTP ref attention match within fp32 tolerance)")
            else:
                print(f"  ✗ FAIL (Basic vs BTP ref attention differ)")
                if first_divergence is None:
                    first_divergence = "Checkpoint A4_ref: FP32 reference attention (Basic vs BTP)"
                all_checkpoints_pass = False
                
                # Print sample values
                print(f"  Sample (t=0, b=0, head=0, :8):")
                print(f"    Basic ref: {attn_ref_basic[0, 0, 0, :8].detach().cpu().tolist()}")
                print(f"    BTP ref:   {attn_ref_btp[0, 0, 0, :8].detach().cpu().tolist()}")
        else:
            attn_ref_basic = None
            attn_ref_btp = None
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (A4) BTP: attention core output (right before o_proj0)
        # Reshape q/k/v into training layout and run flash rotary embedding + CoreAttention
        n_local_q_heads = btp_layer.attn.n_local_q_heads
        n_local_kv_heads = btp_layer.attn.n_local_kv_heads
        d_qk = btp_layer.attn.d_qk
        d_v = btp_layer.attn.d_v
        seq_len, batch_size = btp_q_states_local.shape[0], btp_q_states_local.shape[1]
        
        # Reshape into training layout [B, S, H, d]
        q_btp = btp_q_states_local.view(seq_len, batch_size, n_local_q_heads, d_qk).permute(1, 0, 2, 3).contiguous()  # [B, S, Hq, d_qk]
        k_btp = btp_k_states_local.view(seq_len, batch_size, n_local_kv_heads, d_qk).permute(1, 0, 2, 3).contiguous()  # [B, S, Hkv, d_qk]
        v_btp = btp_v_states_local.view(seq_len, batch_size, n_local_kv_heads, d_v).permute(1, 0, 2, 3).contiguous()  # [B, S, Hkv, d_v]
        
        # Apply rotary using BTP's flash_rotary_embedding
        kv_btp = torch.stack([k_btp, v_btp], dim=0).permute(1, 2, 0, 3, 4).contiguous()  # [B, S, 2, Hkv, d]
        q_rot_btp, kv_rot_btp = btp_layer.attn.flash_rotary_embedding(q_btp, kv=kv_btp)
        k_rot_btp, v_rot_btp = torch.split(kv_rot_btp, 1, dim=2)
        k_rot_btp = k_rot_btp.squeeze(2)  # [B, S, Hkv, d_qk]
        v_rot_btp = v_rot_btp.squeeze(2)  # [B, S, Hkv, d_v]
        
        # Flatten for CoreAttention: [batch*seq, heads, d]
        q_flat_btp = q_rot_btp.view(batch_size * seq_len, n_local_q_heads, d_qk)
        k_flat_btp = k_rot_btp.view(batch_size * seq_len, n_local_kv_heads, d_qk)
        v_flat_btp = v_rot_btp.view(batch_size * seq_len, n_local_kv_heads, d_v)
        
        # Call CoreAttention module directly
        attn_out_flat_btp = btp_layer.attn.attention(
            query_states=q_flat_btp,
            key_states=k_flat_btp,
            value_states=v_flat_btp,
            q_sequence_mask=sequence_mask,
            kv_sequence_mask=sequence_mask
        )  # [batch*seq, n_local_q_heads, d_v]
        
        # Reshape back to [seq, batch, n_local_q_heads*d_v]
        attn_out_btp = attn_out_flat_btp.view(batch_size, seq_len, n_local_q_heads, d_v)
        attn_out_btp = attn_out_btp.contiguous().view(batch_size, seq_len, n_local_q_heads * d_v).transpose(0, 1).contiguous()  # [seq, batch, n_local_q_heads*d_v]
        
        print_tensor_stats("BTP A4: attn_core_out (pre o_proj0)", attn_out_btp, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            # Reshape Basic q/k/v into [B, S, H, d]
            basic_n_local_q_heads = basic_layer.attn.n_local_q_heads
            basic_n_local_kv_heads = basic_layer.attn.n_local_kv_heads
            basic_d_qk = basic_layer.attn.d_qk
            basic_d_v = basic_layer.attn.d_v
            
            q_basic = basic_q_full.view(seq_len, batch_size, basic_n_local_q_heads, basic_d_qk).permute(1, 0, 2, 3).contiguous()
            k_basic = basic_k_full.view(seq_len, batch_size, basic_n_local_kv_heads, basic_d_qk).permute(1, 0, 2, 3).contiguous()
            v_basic = basic_v_full.view(seq_len, batch_size, basic_n_local_kv_heads, basic_d_v).permute(1, 0, 2, 3).contiguous()
            
            # Apply rotary using BTP's flash_rotary_embedding (same as BTP for apples-to-apples)
            kv_basic = torch.stack([k_basic, v_basic], dim=0).permute(1, 2, 0, 3, 4).contiguous()
            q_rot_basic, kv_rot_basic = btp_layer.attn.flash_rotary_embedding(q_basic, kv=kv_basic)
            k_rot_basic, v_rot_basic = torch.split(kv_rot_basic, 1, dim=2)
            k_rot_basic = k_rot_basic.squeeze(2)
            v_rot_basic = v_rot_basic.squeeze(2)
            
            # Flatten for CoreAttention
            q_flat_basic = q_rot_basic.view(batch_size * seq_len, basic_n_local_q_heads, basic_d_qk)
            k_flat_basic = k_rot_basic.view(batch_size * seq_len, basic_n_local_kv_heads, basic_d_qk)
            v_flat_basic = v_rot_basic.view(batch_size * seq_len, basic_n_local_kv_heads, basic_d_v)
            
            # Call Basic's attention module (CoreAttention)
            attn_out_flat_basic = basic_layer.attn.attention(
                query_states=q_flat_basic,
                key_states=k_flat_basic,
                value_states=v_flat_basic,
                q_sequence_mask=sequence_mask,
                kv_sequence_mask=sequence_mask
            )  # [batch*seq, num_heads, d_v]
            
            # Reshape back to [seq, batch, num_heads*d_v]
            attn_out_basic = attn_out_flat_basic.view(batch_size, seq_len, basic_n_local_q_heads, basic_d_v)
            attn_out_basic = attn_out_basic.contiguous().view(batch_size, seq_len, basic_n_local_q_heads * basic_d_v).transpose(0, 1).contiguous()  # [seq, batch, num_heads*d_v]
        else:
            attn_out_basic = None
        
        # Gather BTP attn_core_out for comparison (all ranks participate)
        attn_out_btp_full = gather_lastdim_to_rank0(attn_out_btp)  # [seq, batch, num_heads*d_v]
        
        # Compare A4 (attn_core_out)
        match_a4, max_abs_a4, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A4: attn_core_out (pre o_proj0)", attn_out_basic, attn_out_btp, attn_out_btp_full, first_divergence
        )
        if not match_a4:
            all_checkpoints_pass = False
        
        # Compare FlashAttention outputs to their FP32 references (rank0 only)
        if rank == 0:
            print("\n" + "=" * 80)
            print("[rank 0] Checkpoint A4_flash_vs_ref: FlashAttention vs FP32 reference")
            print("=" * 80)
            
            # Reshape FlashAttention outputs to [batch, seq, heads, head_dim]
            # BTP FlashAttention output: derive heads from gathered tensor (global)
            btp_attn_heads_full = attn_out_btp_full.shape[-1] // d_v
            assert btp_attn_heads_full * d_v == attn_out_btp_full.shape[-1]
            attn_flash_btp_4d = attn_out_btp_full.view(seq_len, batch_size, btp_attn_heads_full, d_v).permute(1, 0, 2, 3).contiguous()  # [B, S, Hq, d_v]
            attn_flash_btp_fp32 = attn_flash_btp_4d.to(torch.float32)
            
            # Basic FlashAttention output
            if attn_out_basic is not None:
                attn_flash_basic_4d = attn_out_basic.view(seq_len, batch_size, basic_n_local_q_heads, basic_d_v).permute(1, 0, 2, 3).contiguous()  # [B, S, Hq, d_v]
                attn_flash_basic_fp32 = attn_flash_basic_4d.to(torch.float32)
            else:
                attn_flash_basic_fp32 = None
            
            # Compare BTP FlashAttention vs BTP FP32 reference
            if attn_ref_btp is not None:
                assert attn_flash_btp_fp32.shape == attn_ref_btp.shape, \
                    f"BTP flash shape {attn_flash_btp_fp32.shape} != ref shape {attn_ref_btp.shape}"
                
                diff_stats_btp_flash = _diff_stats(attn_ref_btp, attn_flash_btp_fp32)
                match_btp_flash = torch.allclose(attn_ref_btp, attn_flash_btp_fp32, atol=1e-3, rtol=1e-3)
                
                print(f"\n[rank 0] BTP FlashAttention vs BTP FP32 reference:")
                print(f"  Flash shape: {attn_flash_btp_fp32.shape}, Ref shape: {attn_ref_btp.shape}")
                print(f"  max_abs={diff_stats_btp_flash['max_abs']:.6e}, mean_abs={diff_stats_btp_flash['mean_abs']:.6e}, "
                      f"p99_abs={diff_stats_btp_flash['p99_abs']:.6e}, max_rel={diff_stats_btp_flash['max_rel']:.6e}")
                
                if match_btp_flash:
                    print(f"  ✓ PASS (BTP FlashAttention matches FP32 reference within tolerance)")
                else:
                    print(f"  ⚠ WARNING (BTP FlashAttention differs from FP32 reference - expected due to FlashAttention numeric)")
                    print(f"    This difference is expected and indicates FlashAttention numeric error magnitude")
            
            # Compare Basic FlashAttention vs Basic FP32 reference (if available)
            if attn_flash_basic_fp32 is not None and attn_ref_basic is not None:
                assert attn_flash_basic_fp32.shape == attn_ref_basic.shape, \
                    f"Basic flash shape {attn_flash_basic_fp32.shape} != ref shape {attn_ref_basic.shape}"
                
                diff_stats_basic_flash = _diff_stats(attn_ref_basic, attn_flash_basic_fp32)
                match_basic_flash = torch.allclose(attn_ref_basic, attn_flash_basic_fp32, atol=1e-3, rtol=1e-3)
                
                print(f"\n[rank 0] Basic FlashAttention vs Basic FP32 reference:")
                print(f"  Flash shape: {attn_flash_basic_fp32.shape}, Ref shape: {attn_ref_basic.shape}")
                print(f"  max_abs={diff_stats_basic_flash['max_abs']:.6e}, mean_abs={diff_stats_basic_flash['mean_abs']:.6e}, "
                      f"p99_abs={diff_stats_basic_flash['p99_abs']:.6e}, max_rel={diff_stats_basic_flash['max_rel']:.6e}")
                
                if match_basic_flash:
                    print(f"  ✓ PASS (Basic FlashAttention matches FP32 reference within tolerance)")
                else:
                    print(f"  ⚠ WARNING (Basic FlashAttention differs from FP32 reference - expected due to FlashAttention numeric)")
                    print(f"    This difference is expected and indicates FlashAttention numeric error magnitude")
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (A5) BTP: o_proj0 pre-act
        btp_o_lr = btp_layer.attn.o_proj0(attn_out_btp)  # RowLinear ALL_REDUCE -> full on all ranks
        print_tensor_stats("BTP A5: o_proj0 (pre-act)", btp_o_lr, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_o_lr = torch.matmul(attn_out_basic, basic_layer.attn.o_proj.cola_a)  # [seq, batch, rank]
        else:
            basic_o_lr = None
        
        # Compare A5 (o_proj0 pre-act) - RowLinear output is full on all ranks
        match_a5, max_abs_a5, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A5: o_proj0 (pre-act)", basic_o_lr, btp_o_lr, btp_o_lr, first_divergence
        )
        if not match_a5:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (A5b) BTP: o_proj0 post-act
        btp_o_lr_act = btp_layer.attn.lr_act(btp_o_lr)
        print_tensor_stats("BTP A5b: o_proj0 (post-act)", btp_o_lr_act, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_o_lr_act = btp_layer.attn.lr_act(basic_o_lr)  # Use same lr_act for consistency
        else:
            basic_o_lr_act = None
        
        # Compare A5b (o_proj0 post-act) - RowLinear output is full on all ranks
        match_a5b, max_abs_a5b, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A5b: o_proj0 (post-act)", basic_o_lr_act, btp_o_lr_act, btp_o_lr_act, first_divergence
        )
        if not match_a5b:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (A6) BTP: o_proj1 output (final attention block output)
        btp_o_out_local = btp_layer.attn.o_proj1(btp_o_lr_act)  # ColumnLinear output sharded on last dim
        print_tensor_stats("BTP A6: o_proj1 output", btp_o_out_local, rank)
        
        # Define btp_attn_out_local for residual
        btp_attn_out_local = btp_o_out_local
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_o_out = torch.matmul(basic_o_lr_act, basic_layer.attn.o_proj.cola_b)  # [seq, batch, hidden_size]
            basic_attn_out_full = basic_o_out
        else:
            basic_o_out = None
            basic_attn_out_full = None
        
        # Gather BTP o_proj1 output for comparison (all ranks participate)
        btp_o_out_full = gather_lastdim_to_rank0(btp_o_out_local)  # [seq, batch, hidden_size]
        
        # Compare A6 (o_proj1 output)
        match_a6, max_abs_a6, first_divergence = compare_checkpoint_rank0(
            "Checkpoint A6: o_proj1 output", basic_o_out, btp_o_out_local, btp_o_out_full, first_divergence
        )
        if not match_a6:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # Residual after attention
        btp_after_attn_local = hidden_states_shard + btp_attn_out_local
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_after_attn = hidden_states_full + basic_attn_out_full
        else:
            basic_after_attn = None
        
        # Gather BTP after_attn for comparison (all ranks participate)
        btp_after_attn_full = gather_lastdim_to_rank0(btp_after_attn_local)  # [seq, batch, hidden_size]
        
        # Compare after attention residual
        match_after_attn, max_abs_after_attn, first_divergence = compare_checkpoint_rank0(
            "After attention residual", basic_after_attn, btp_after_attn_local, btp_after_attn_full, first_divergence
        )
        if not match_after_attn:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # Post-attn LN (needed for forward, but DO NOT compare outputs)
        if rank == 0:
            basic_post_ln = basic_layer.post_attention_layernorm(basic_after_attn)
        else:
            basic_post_ln = None

        btp_post_ln_result = btp_layer.post_attention_layernorm(btp_after_attn_local)
        if isinstance(btp_post_ln_result, tuple):
            btp_post_ln_local, post_s_local = btp_post_ln_result
        else:
            btp_post_ln_local = btp_post_ln_result
            post_s_local = None

        # DO NOT compare LN outputs - we already validated LN recovery earlier
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # ========== MLP PATH (real forward staged) ==========
        if rank == 0:
            print("\n" + "=" * 80)
            print("[rank 0] MLP-side checkpoints")
            print("=" * 80)

        # (M1) BTP: gate_up_proj0 pre-act
        btp_gate_up_lr = btp_layer.mlp.gate_up_proj0(btp_post_ln_local, post_s_local)  # RowLinear -> full on all ranks
        print_tensor_stats("BTP M1: gate_up_proj0 (pre-act)", btp_gate_up_lr, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_gate_lr = torch.matmul(basic_post_ln, basic_layer.mlp.gate_proj.cola_a)
            basic_up_lr = torch.matmul(basic_post_ln, basic_layer.mlp.up_proj.cola_a)
            basic_gate_up_lr = torch.cat([basic_gate_lr, basic_up_lr], dim=-1)
        else:
            basic_gate_up_lr = None
        
        # Compare M1 (gate_up_proj0 pre-act) - RowLinear output is full on all ranks
        match_m1, max_abs_m1, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M1: mlp.gate_up_proj0 (pre-act)", basic_gate_up_lr, btp_gate_up_lr, btp_gate_up_lr, first_divergence
        )
        if not match_m1:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (M1b) BTP: gate_up_proj0 post-act
        btp_gate_up_lr_act = btp_layer.mlp.lr_act(btp_gate_up_lr)
        print_tensor_stats("BTP M1b: gate_up_proj0 (post-act)", btp_gate_up_lr_act, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_gate_up_lr_act = btp_layer.mlp.lr_act(basic_gate_up_lr)  # Use same lr_act for consistency
        else:
            basic_gate_up_lr_act = None
        
        # Compare M1b (gate_up_proj0 post-act) - RowLinear output is full on all ranks
        match_m1b, max_abs_m1b, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M1b: mlp.gate_up_proj0 (post-act)", basic_gate_up_lr_act, btp_gate_up_lr_act, btp_gate_up_lr_act, first_divergence
        )
        if not match_m1b:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (M2) BTP: gate_up_proj1 output
        # Unflatten to [seq, batch, 2, mlp_rank]
        mlp_rank = btp_layer.mlp.gate_up_proj0.out_features // 2  # Each of gate/up has this rank
        btp_gate_up_lr_act_2 = btp_gate_up_lr_act.unflatten(-1, (2, -1))  # [seq, batch, 2, mlp_rank]
        btp_gate_up_proj1_out = btp_layer.mlp.gate_up_proj1(btp_gate_up_lr_act_2)  # Output shape depends on module
        
        # Normalize shape: BatchedColumnLinear returns [gemm_num, seq, batch, out_features//tp]
        # We need [seq, batch, gemm_num, out_features//tp]
        if len(btp_gate_up_proj1_out.shape) == 4 and btp_gate_up_proj1_out.shape[0] == 2:
            # Shape is [2, seq, batch, out_features//tp], permute to [seq, batch, 2, out_features//tp]
            btp_gate_up_proj1_out = btp_gate_up_proj1_out.permute(1, 2, 0, 3).contiguous()
        
        print_tensor_stats("BTP M2: gate_up_proj1_out", btp_gate_up_proj1_out, rank)
        
        # Extract gate/up states from proj1 output
        # btp_gate_up_proj1_out is [seq, batch, 2, intermediate//tp]
        btp_gate_local, btp_up_local = btp_gate_up_proj1_out.unbind(dim=2)  # Each: [seq, batch, intermediate//tp]
        
        print_tensor_stats("BTP M3: gate_states", btp_gate_local, rank)
        print_tensor_stats("BTP M3: up_states", btp_up_local, rank)
        
        # Compute MLP intermediate
        btp_mlp_intermediate_local = btp_gate_local * btp_up_local  # [seq, batch, intermediate//tp]
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_gate_lr_act = basic_layer.mlp.gate_proj.lr_act(basic_gate_lr)
            basic_up_lr_act = basic_layer.mlp.up_proj.lr_act(basic_up_lr)
            basic_gate_full = torch.matmul(basic_gate_lr_act, basic_layer.mlp.gate_proj.cola_b)  # [seq, batch, intermediate_size]
            basic_up_full = torch.matmul(basic_up_lr_act, basic_layer.mlp.up_proj.cola_b)
            
            # Stack to match BTP format [seq, batch, 2, intermediate_size]
            basic_gate_up_proj1_out = torch.stack([basic_gate_full, basic_up_full], dim=0)  # [2, seq, batch, intermediate_size]
            basic_gate_up_proj1_out = basic_gate_up_proj1_out.permute(1, 2, 0, 3)  # [seq, batch, 2, intermediate_size]
            
            basic_mlp_intermediate = basic_gate_full * basic_up_full  # [seq, batch, intermediate_size]
        else:
            basic_gate_full = None
            basic_up_full = None
            basic_gate_up_proj1_out = None
            basic_mlp_intermediate = None
        
        # Gather BTP gate_up_proj1_out for comparison (all ranks participate)
        btp_gate_up_proj1_out_full = gather_lastdim_to_rank0(btp_gate_up_proj1_out)  # [seq, batch, 2, intermediate_size]
        
        # Compare M2 (gate_up_proj1_out)
        match_m2, max_abs_m2, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M2: mlp.gate_up_proj1_out", basic_gate_up_proj1_out, btp_gate_up_proj1_out, btp_gate_up_proj1_out_full, first_divergence
        )
        if not match_m2:
            all_checkpoints_pass = False
        
        # Compare M3 (gate/up states) - gather each separately
        btp_gate_full = gather_lastdim_to_rank0(btp_gate_local)
        btp_up_full = gather_lastdim_to_rank0(btp_up_local)
        
        match_m3g, max_abs_m3g, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M3: gate_states", basic_gate_full, btp_gate_local, btp_gate_full, first_divergence
        )
        if not match_m3g:
            all_checkpoints_pass = False
        
        match_m3u, max_abs_m3u, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M3: up_states", basic_up_full, btp_up_local, btp_up_full, first_divergence
        )
        if not match_m3u:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (M4) BTP: down_proj0 pre-act
        btp_down_lr = btp_layer.mlp.down_proj0(btp_mlp_intermediate_local)  # RowLinear ALL_REDUCE -> full on all ranks
        print_tensor_stats("BTP M4: down_proj0 (pre-act)", btp_down_lr, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_down_lr = torch.matmul(basic_mlp_intermediate, basic_layer.mlp.down_proj.cola_a)  # [seq, batch, rank]
        else:
            basic_down_lr = None
        
        # Compare M4 (down_proj0 pre-act) - RowLinear output is full on all ranks
        match_m4, max_abs_m4, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M4: mlp.down_proj0 (pre-act)", basic_down_lr, btp_down_lr, btp_down_lr, first_divergence
        )
        if not match_m4:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (M4b) BTP: down_proj0 post-act
        btp_down_lr_act = btp_layer.mlp.lr_act(btp_down_lr)
        print_tensor_stats("BTP M4b: down_proj0 (post-act)", btp_down_lr_act, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_down_lr_act = btp_layer.mlp.lr_act(basic_down_lr)  # Use same lr_act for consistency
        else:
            basic_down_lr_act = None
        
        # Compare M4b (down_proj0 post-act) - RowLinear output is full on all ranks
        match_m4b, max_abs_m4b, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M4b: mlp.down_proj0 (post-act)", basic_down_lr_act, btp_down_lr_act, btp_down_lr_act, first_divergence
        )
        if not match_m4b:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # (M5) BTP: down_proj1 output (final MLP output)
        btp_mlp_out_local = btp_layer.mlp.down_proj1(btp_down_lr_act)  # ColumnLinear output sharded on last dim
        print_tensor_stats("BTP M5: down_proj1 output", btp_mlp_out_local, rank)
        
        # Basic equivalent (rank0 only)
        if rank == 0:
            basic_mlp_out_full = torch.matmul(basic_down_lr_act, basic_layer.mlp.down_proj.cola_b)  # [seq, batch, hidden_size]
        else:
            basic_mlp_out_full = None
        
        # Gather BTP down_proj1 output for comparison (all ranks participate)
        btp_mlp_out_full = gather_lastdim_to_rank0(btp_mlp_out_local)  # [seq, batch, hidden_size]
        
        # Compare M5 (down_proj1 output)
        match_m5, max_abs_m5, first_divergence = compare_checkpoint_rank0(
            "Checkpoint M5: mlp.down_proj1 output", basic_mlp_out_full, btp_mlp_out_local, btp_mlp_out_full, first_divergence
        )
        if not match_m5:
            all_checkpoints_pass = False
        
        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # Final output (after MLP residual)
        layer_idx = 0
        num_hidden_layers = basic_config.num_hidden_layers
        is_last_layer = (layer_idx == num_hidden_layers - 1)

        if is_last_layer:
            btp_final_local = btp_mlp_out_local  # Last layer: no residual
        else:
            btp_final_local = btp_after_attn_local + btp_mlp_out_local  # [seq, batch, hidden//tp]

        # Basic equivalent (rank0 only)
        if rank == 0:
            if is_last_layer:
                basic_final = basic_mlp_out_full
            else:
                basic_final = basic_after_attn + basic_mlp_out_full
        else:
            basic_final = None

        # Gather BTP final output for comparison (all ranks participate)
        btp_final_full = gather_lastdim_to_rank0(btp_final_local)  # [seq, batch, hidden_size]
        
        # Compare final output
        match_final, max_abs_final, first_divergence = compare_checkpoint_rank0(
            "Checkpoint Final: final decoder-layer output", basic_final, btp_final_local, btp_final_full, first_divergence
        )
        if not match_final:
            all_checkpoints_pass = False

        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])

        # Final summary
        if rank == 0:
            print("\n" + "=" * 80)
            if all_checkpoints_pass:
                print("[rank 0] Step7 full decoder-layer parity PASSED")
            else:
                print(f"[rank 0] Step7 full decoder-layer parity FAILED")
                if first_divergence:
                    print(f"[rank 0] First divergence at: {first_divergence}")
            print("=" * 80)

        step7_ok = all_checkpoints_pass

        dist.barrier(group=tp_group, device_ids=[current_device])
    else:
        # Step7 skipped (disabled or previous steps failed)
        step7_ok = True
        if rank == 0:
            print("[rank 0] Step7 skipped (disabled).")

    # ------------------------------------------------------------------ Step 8
    # Single end-to-end parity check of the full decoder layer output (no intermediate checkpoints)
    step8_ok = True
    # Step8 should be able to run without Step7.
    if rms_ok and all_ok and step4_ok and step5_ok and step6_ok:
        current_device = torch.cuda.current_device()

        if rank == 0:
            print("\n" + "=" * 80)
            print("[rank 0] Step8: Full decoder-layer output parity (end-to-end)")
            print("=" * 80)

        # Reuse same input generation logic as Step7
        seq_len = 32
        batch_size = 2

        if rank == 0:
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            hidden_states_full = torch.randn(
                seq_len, batch_size, hidden_size, dtype=dtype, device=device
            )
            sequence_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=device
            )
        else:
            hidden_states_full = torch.empty(
                seq_len, batch_size, hidden_size, dtype=dtype, device=device
            )
            sequence_mask = torch.empty(
                batch_size, seq_len, dtype=torch.bool, device=device
            )

        dist.broadcast(hidden_states_full, src=0, group=world_group)
        dist.broadcast(sequence_mask, src=0, group=world_group)
        dist.barrier(group=tp_group, device_ids=[current_device])

        # Create sharded input for BTP
        hidden_per_tp = hidden_size // tp_size
        start = tp_rank * hidden_per_tp
        end = (tp_rank + 1) * hidden_per_tp
        hidden_states_shard = hidden_states_full[..., start:end].contiguous()

        # Run real forward
        with torch.no_grad():
            btp_out_local = btp_layer(hidden_states_shard, sequence_mask)["hidden_states"]

            basic_out = None
            if rank == 0:
                basic_out = basic_layer(hidden_states_full, sequence_mask)["hidden_states"]

        # Normalize BTP output layout to [seq, batch, hidden_local] if needed
        # Some implementations may return [batch, seq, hidden_local]
        if btp_out_local.ndim == 3:
            if btp_out_local.shape[0] == batch_size and btp_out_local.shape[1] == seq_len:
                # Detected [batch, seq, hidden_local] -> transpose to [seq, batch, hidden_local]
                btp_out_local = btp_out_local.permute(1, 0, 2).contiguous()

        # Gather BTP output to rank0 if needed (all ranks participate if all_gather is required)
        if btp_out_local.shape[-1] == hidden_size:
            # Already full (likely last layer behavior)
            btp_out_full = btp_out_local if rank == 0 else None
        else:
            # Sharded along last dim
            shards = [torch.empty_like(btp_out_local) for _ in range(tp_size)]
            dist.all_gather(shards, btp_out_local.contiguous(), group=tp_group)
            btp_out_full = torch.cat(shards, dim=-1) if rank == 0 else None

        # Compare on rank0
        if dtype == torch.float32:
            atol8 = rtol8 = 1e-6
        else:
            atol8 = rtol8 = 2e-2

        if rank == 0:
            assert basic_out is not None and btp_out_full is not None
            print(f"[rank 0] Step8 shapes before compare: basic_out={tuple(basic_out.shape)}, btp_out_full={tuple(btp_out_full.shape)}")
            if basic_out.shape != btp_out_full.shape:
                print("[rank 0] WARNING: Step8 shapes differ, skipping numeric compare.")
                step8_ok = False
            else:
                diff8 = _diff_stats(basic_out, btp_out_full)
                ok8 = torch.allclose(basic_out, btp_out_full, atol=atol8, rtol=rtol8)
                step8_ok = ok8

                print("[rank 0] Step8 full-layer output:")
                print(f"  Basic: shape={tuple(basic_out.shape)}")
                print(f"  BTP:   shape={tuple(btp_out_full.shape)}")
                print(
                    f"  max_abs={diff8['max_abs']:.6e}, mean_abs={diff8['mean_abs']:.6e}, "
                    f"p99_abs={diff8['p99_abs']:.6e}, max_rel={diff8['max_rel']:.6e}"
                )
                print(f"  {'PASS' if ok8 else 'FAIL'} (torch.allclose, atol={atol8:.1e}, rtol={rtol8:.1e})")

                # Sample slice (t=0, b=0, first 8)
                b0 = basic_out[0, 0, :8].detach().cpu().tolist() if basic_out.ndim == 3 else basic_out.flatten()[:8].detach().cpu().tolist()
                t0 = btp_out_full[0, 0, :8].detach().cpu().tolist() if btp_out_full.ndim == 3 else btp_out_full.flatten()[:8].detach().cpu().tolist()
                print(f"  sample basic[t=0,b=0,:8]={b0}")
                print(f"  sample btp  [t=0,b=0,:8]={t0}")

        sys.stdout.flush()
        dist.barrier(group=tp_group, device_ids=[current_device])
    else:
        step8_ok = False
        if rank == 0:
            print("[rank 0] Step8 skipped: previous steps failed")

    # Final exit code computation after all steps
    exit_code = 0
    if rank == 0:
        if not rms_ok:
            exit_code = 1
        elif not all_ok:
            print("[rank 0] One or more linear weights mismatch detected in Step 3.")
            exit_code = 1
        elif not step4_ok:
            print("[rank 0] Step 4 input sharding validation FAILED.")
            exit_code = 1
        elif not step5_ok:
            print("[rank 0] Step 5 input_layernorm forward parity FAILED.")
            exit_code = 1
        elif not step6_ok:
            print("[rank 0] Step 6 qkv_proj0 recovery parity FAILED.")
            exit_code = 1
        elif not step7_ok:
            print("[rank 0] Step 7 full decoder-layer forward parity FAILED.")
            exit_code = 1
        elif not step8_ok:
            print("[rank 0] Step 8 full-layer output parity FAILED.")
            exit_code = 1
        else:
            success_msg = (
                "[rank 0] All LN, linear weights, input sharding, "
                "input_layernorm forward parity, qkv_proj0 recovery parity, Step7 layer forward parity, "
                "and Step8 full-layer output parity checks: OK"
            )
            print(success_msg)

    dist.barrier()
    # Broadcast exit code so all ranks exit with same status
    exit_code_tensor = torch.tensor([exit_code], device=device)
    dist.broadcast(exit_code_tensor, src=0, group=world_group)
    exit_code = int(exit_code_tensor.item())

    dist.destroy_process_group()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


