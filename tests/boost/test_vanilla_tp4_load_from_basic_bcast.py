"""
Test to verify forward compute parity between BasicCoLA (TP=1) and VanillaCoLA (TP=4).

Weight-loading strategy:
1. Rank 0 creates BasicCoLA decoder layer (TP=1) as ground-truth weight source
2. Broadcast FULL Basic layer weights from rank 0 to all TP ranks
3. Each TP rank builds VanillaCoLA decoder layer (TP=4) and loads ONLY its local shard
   of those Basic weights (different slice per rank, according to TP sharding rules)
4. Run forward with identical inputs and compare gathered Vanilla output to Basic output

Run with: torchrun --nproc_per_node=4 test_vanilla_tp4_load_from_basic_bcast.py
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

# Add examples/cola to path
cola_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "cola")
sys.path.insert(0, cola_dir)

from config_basic_cola_llama import BasicColaLlamaConfig
from config_cola_llama import ColaLlamaConfig
from basic_cola_llama import BasicColaLlamaDecoderLayer
from vanilla_cola_llama import LlamaDecoderLayer as VanillaLlamaDecoderLayer

from nanotron.config import ParallelismArgs
from nanotron.parallel import ParallelContext


def create_test_configs():
    """Create matching configs for Basic and Vanilla models"""
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

    vanilla_config = ColaLlamaConfig(
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

    return basic_config, vanilla_config


def slice_for_param(full_tensor: torch.Tensor, shard_dim: int, rank: int, tp_size: int) -> torch.Tensor:
    """
    Extract the local shard slice from a full tensor based on TP sharding rules.
    
    Args:
        full_tensor: Full weight tensor [..., dim_to_shard, ...]
        shard_dim: Dimension to shard (0 for output dim, 1 for input dim)
        rank: Current TP rank
        tp_size: Tensor parallel size
    
    Returns:
        Local shard slice
    """
    dim_size = full_tensor.shape[shard_dim]
    assert dim_size % tp_size == 0, f"Dimension {shard_dim} size {dim_size} must be divisible by tp_size {tp_size}"
    
    shard_size = dim_size // tp_size
    start_idx = rank * shard_size
    end_idx = (rank + 1) * shard_size
    
    if shard_dim == 0:
        # Shard output dimension (first dim): full[out, in] -> local[out//tp, in]
        return full_tensor[start_idx:end_idx, :]
    elif shard_dim == 1:
        # Shard input dimension (second dim): full[out, in] -> local[out, in//tp]
        return full_tensor[:, start_idx:end_idx]
    else:
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}")


def broadcast_tensor(tensor: torch.Tensor, src_rank: int, group, rank: int) -> torch.Tensor:
    """
    Broadcast a tensor from src_rank to all ranks in the group.
    
    Args:
        tensor: Tensor to broadcast (on src_rank) or empty tensor (on other ranks)
        src_rank: Source rank
        group: Process group
        rank: Current rank
    
    Returns:
        Broadcasted tensor (same on all ranks)
    """
    if rank != src_rank:
        # Other ranks: tensor should be pre-allocated with correct shape/dtype/device
        pass
    dist.broadcast(tensor, src=src_rank, group=group)
    return tensor


def load_vanilla_weights_from_broadcasted_basic(
    vanilla_layer,
    # Broadcasted Basic weights (full tensors, same on all ranks after broadcast)
    input_layernorm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    # Attention weights
    q_cola_a_full: torch.Tensor,  # [hidden_size, rank]
    k_cola_a_full: torch.Tensor,  # [hidden_size, rank]
    v_cola_a_full: torch.Tensor,  # [hidden_size, rank]
    q_cola_b_full: torch.Tensor,  # [rank, out_features]
    k_cola_b_full: torch.Tensor,  # [rank, out_features]
    v_cola_b_full: torch.Tensor,  # [rank, out_features]
    o_cola_a_full: torch.Tensor,  # [hidden_size, rank]
    o_cola_b_full: torch.Tensor,  # [rank, hidden_size]
    # MLP weights
    gate_cola_a_full: torch.Tensor,  # [hidden_size, rank]
    up_cola_a_full: torch.Tensor,  # [hidden_size, rank]
    gate_cola_b_full: torch.Tensor,  # [rank, intermediate_size]
    up_cola_b_full: torch.Tensor,  # [rank, intermediate_size]
    down_cola_a_full: torch.Tensor,  # [intermediate_size, rank]
    down_cola_b_full: torch.Tensor,  # [rank, hidden_size]
    rank: int,
    tp_size: int,
):
    """
    Load Vanilla TP=4 layer weights from broadcasted full Basic tensors.
    Each rank extracts its local shard based on TP sharding rules.
    """
    with torch.no_grad():
        # Layer norms: not sharded, every rank copies full weight
        vanilla_layer.input_layernorm.weight.copy_(input_layernorm_weight)
        vanilla_layer.post_attention_layernorm.weight.copy_(post_attention_layernorm_weight)
        
        # Attention QKV proj0 (ColumnLinear: shards output dim)
        # IMPORTANT: Each rank should get [Q_shard; K_shard; V_shard], not a contiguous slice
        # For rank r with tp_size, each gets rank//tp_size rows of Q, K, V
        q_cola_a_t = q_cola_a_full.t()  # [rank, hidden_size] = [64, 128]
        k_cola_a_t = k_cola_a_full.t()  # [rank, hidden_size] = [64, 128]
        v_cola_a_t = v_cola_a_full.t()  # [rank, hidden_size] = [64, 128]
        
        # Get attn_rank from tensor shape
        attn_rank = q_cola_a_t.shape[0]  # Should be 64
        
        # Shard each Q, K, V separately, then concatenate per rank
        rank_per_tp = attn_rank // tp_size  # 64 // 4 = 16
        q_shard = q_cola_a_t[rank * rank_per_tp:(rank + 1) * rank_per_tp, :]  # [16, 128]
        k_shard = k_cola_a_t[rank * rank_per_tp:(rank + 1) * rank_per_tp, :]  # [16, 128]
        v_shard = v_cola_a_t[rank * rank_per_tp:(rank + 1) * rank_per_tp, :]  # [16, 128]
        
        # Concatenate per-rank: [Q_shard; K_shard; V_shard] = [48, 128]
        local_qkv_proj0 = torch.cat([q_shard, k_shard, v_shard], dim=0)  # [3*rank_per_tp, hidden_size] = [48, 128]
        
        # Verify shape matches expected
        expected_shape = vanilla_layer.attn.qkv_proj0.weight.shape
        assert local_qkv_proj0.shape == expected_shape, \
            f"qkv_proj0 shape mismatch: got {local_qkv_proj0.shape}, expected {expected_shape}"
        vanilla_layer.attn.qkv_proj0.weight.copy_(local_qkv_proj0)
        
        # Q/K/V proj1 (RowLinear: shards input dim)
        for proj_name, cola_b_full in [("q", q_cola_b_full), ("k", k_cola_b_full), ("v", v_cola_b_full)]:
            full_proj1 = cola_b_full.t()  # [out_features, rank]
            # RowLinear shards input dim (dim 1)
            local_proj1 = slice_for_param(full_proj1, shard_dim=1, rank=rank, tp_size=tp_size)
            vanilla_proj1 = getattr(vanilla_layer.attn, f"{proj_name}_proj1")
            expected_shape = vanilla_proj1.weight.shape
            assert local_proj1.shape == expected_shape, \
                f"{proj_name}_proj1 shape mismatch: got {local_proj1.shape}, expected {expected_shape}"
            vanilla_proj1.weight.copy_(local_proj1)
        
        # O proj0 (ColumnLinear: shards output dim)
        full_o_proj0 = o_cola_a_full.t()  # [rank, hidden_size]
        local_o_proj0 = slice_for_param(full_o_proj0, shard_dim=0, rank=rank, tp_size=tp_size)
        assert local_o_proj0.shape == vanilla_layer.attn.o_proj0.weight.shape, \
            f"o_proj0 shape mismatch: got {local_o_proj0.shape}, expected {vanilla_layer.attn.o_proj0.weight.shape}"
        vanilla_layer.attn.o_proj0.weight.copy_(local_o_proj0)
        
        # O proj1 (RowLinear: shards input dim)
        full_o_proj1 = o_cola_b_full.t()  # [hidden_size, rank]
        local_o_proj1 = slice_for_param(full_o_proj1, shard_dim=1, rank=rank, tp_size=tp_size)
        assert local_o_proj1.shape == vanilla_layer.attn.o_proj1.weight.shape, \
            f"o_proj1 shape mismatch: got {local_o_proj1.shape}, expected {vanilla_layer.attn.o_proj1.weight.shape}"
        vanilla_layer.attn.o_proj1.weight.copy_(local_o_proj1)
        
        # MLP gate_up_proj0 (ColumnLinear: shards output dim)
        # IMPORTANT: Each rank should get [gate_shard; up_shard], not a contiguous slice
        gate_cola_a_t = gate_cola_a_full.t()  # [rank, hidden_size] = [64, 128]
        up_cola_a_t = up_cola_a_full.t()  # [rank, hidden_size] = [64, 128]
        
        # Get mlp_rank from tensor shape
        mlp_rank = gate_cola_a_t.shape[0]  # Should be 64
        
        # Shard each gate, up separately, then concatenate per rank
        rank_per_tp = mlp_rank // tp_size  # 64 // 4 = 16
        gate_shard = gate_cola_a_t[rank * rank_per_tp:(rank + 1) * rank_per_tp, :]  # [16, 128]
        up_shard = up_cola_a_t[rank * rank_per_tp:(rank + 1) * rank_per_tp, :]  # [16, 128]
        
        # Concatenate per-rank: [gate_shard; up_shard] = [32, 128]
        local_gate_up_proj0 = torch.cat([gate_shard, up_shard], dim=0)  # [2*rank_per_tp, hidden_size] = [32, 128]
        assert local_gate_up_proj0.shape == vanilla_layer.mlp.gate_up_proj0.weight.shape, \
            f"gate_up_proj0 shape mismatch: got {local_gate_up_proj0.shape}, expected {vanilla_layer.mlp.gate_up_proj0.weight.shape}"
        vanilla_layer.mlp.gate_up_proj0.weight.copy_(local_gate_up_proj0)
        
        # Gate/Up proj1 (RowLinear: shards input dim)
        for proj_name, cola_b_full in [("gate", gate_cola_b_full), ("up", up_cola_b_full)]:
            full_proj1 = cola_b_full.t()  # [intermediate_size, rank]
            local_proj1 = slice_for_param(full_proj1, shard_dim=1, rank=rank, tp_size=tp_size)
            vanilla_proj1 = getattr(vanilla_layer.mlp, f"{proj_name}_proj1")
            assert local_proj1.shape == vanilla_proj1.weight.shape, \
                f"{proj_name}_proj1 shape mismatch: got {local_proj1.shape}, expected {vanilla_proj1.weight.shape}"
            vanilla_proj1.weight.copy_(local_proj1)
        
        # Down proj0 (ColumnLinear: shards output dim)
        full_down_proj0 = down_cola_a_full.t()  # [rank, intermediate_size]
        local_down_proj0 = slice_for_param(full_down_proj0, shard_dim=0, rank=rank, tp_size=tp_size)
        assert local_down_proj0.shape == vanilla_layer.mlp.down_proj0.weight.shape, \
            f"down_proj0 shape mismatch: got {local_down_proj0.shape}, expected {vanilla_layer.mlp.down_proj0.weight.shape}"
        vanilla_layer.mlp.down_proj0.weight.copy_(local_down_proj0)
        
        # Down proj1 (RowLinear: shards input dim)
        full_down_proj1 = down_cola_b_full.t()  # [hidden_size, rank]
        local_down_proj1 = slice_for_param(full_down_proj1, shard_dim=1, rank=rank, tp_size=tp_size)
        assert local_down_proj1.shape == vanilla_layer.mlp.down_proj1.weight.shape, \
            f"down_proj1 shape mismatch: got {local_down_proj1.shape}, expected {vanilla_layer.mlp.down_proj1.weight.shape}"
        vanilla_layer.mlp.down_proj1.weight.copy_(local_down_proj1)


def gather_vanilla_weight_shard(weight_shard: torch.Tensor, full_shape: Tuple[int, ...], tp_pg, rank: int, tp_size: int) -> torch.Tensor:
    """
    Gather sharded weight from all TP ranks to reconstruct full weight tensor on rank 0.
    
    Args:
        weight_shard: Local shard [out_features//tp, in_features] or [out_features, in_features//tp]
        full_shape: Full weight shape [out_features, in_features]
        tp_pg: Tensor parallel process group
        rank: Current rank
        tp_size: Tensor parallel size
    
    Returns:
        Full weight tensor on rank 0, None on other ranks
    """
    # Determine which dimension is sharded
    if weight_shard.shape[0] < full_shape[0]:
        # Output dim is sharded (ColumnLinear)
        shard_dim = 0
        shard_size = full_shape[0] // tp_size
    elif weight_shard.shape[1] < full_shape[1]:
        # Input dim is sharded (RowLinear)
        shard_dim = 1
        shard_size = full_shape[1] // tp_size
    else:
        # Not sharded
        if rank == 0:
            return weight_shard
        else:
            return None
    
    # Create full tensor on rank 0
    if rank == 0:
        full_weight = torch.empty(full_shape, dtype=weight_shard.dtype, device=weight_shard.device)
    else:
        full_weight = None
    
    # Gather shards (all ranks must participate)
    shard_list = [torch.empty_like(weight_shard) for _ in range(tp_size)]
    dist.all_gather(shard_list, weight_shard.contiguous(), group=tp_pg)
    
    # Reconstruct full weight (only rank 0 needs it, but all ranks participate in gather)
    if rank == 0:
        if shard_dim == 0:
            # Shards are along output dim, concatenate along dim 0
            full_weight = torch.cat(shard_list, dim=0)
        else:
            # Shards are along input dim, concatenate along dim 1
            full_weight = torch.cat(shard_list, dim=1)
        return full_weight
    else:
        return None


def gather_vanilla_grouped_weight_shard(weight_shard: torch.Tensor, full_shape: Tuple[int, ...], num_groups: int, tp_pg, rank: int, tp_size: int) -> torch.Tensor:
    """
    Gather sharded grouped weight (e.g., qkv_proj0 with [Q;K;V] or gate_up_proj0 with [gate;up]).
    
    Each rank has layout [group0_shard; group1_shard; ...] where each group is sharded.
    After gathering, we need to reorder to [group0_full; group1_full; ...].
    
    Args:
        weight_shard: Local shard [groups*rank_per_tp, in_features] = [48, 128] for qkv
        full_shape: Full weight shape [groups*rank, in_features] = [192, 128] for qkv
        num_groups: Number of groups (3 for QKV, 2 for gate_up)
        tp_pg: Tensor parallel process group
        rank: Current rank
        tp_size: Tensor parallel size
    
    Returns:
        Full weight tensor on rank 0, None on other ranks
    """
    # Gather shards (all ranks must participate)
    shard_list = [torch.empty_like(weight_shard) for _ in range(tp_size)]
    dist.all_gather(shard_list, weight_shard.contiguous(), group=tp_pg)
    
    if rank == 0:
        # Each shard has shape [groups*rank_per_tp, in_features]
        # Shard from rank r has: [group0_r_shard; group1_r_shard; ...]
        rank_per_tp = weight_shard.shape[0] // num_groups  # 48 // 3 = 16 for qkv
        
        # Extract and reorder: collect all group0 shards, then all group1 shards, etc.
        full_weight = torch.empty(full_shape, dtype=weight_shard.dtype, device=weight_shard.device)
        
        for group_idx in range(num_groups):
            # Extract group_idx shard from each rank
            group_shards = []
            for r in range(tp_size):
                start_row = group_idx * rank_per_tp
                end_row = (group_idx + 1) * rank_per_tp
                group_shard = shard_list[r][start_row:end_row, :]  # [rank_per_tp, in_features]
                group_shards.append(group_shard)
            
            # Concatenate all ranks' shards for this group
            group_full = torch.cat(group_shards, dim=0)  # [rank, in_features]
            
            # Place in full_weight at correct position
            full_start_row = group_idx * (full_shape[0] // num_groups)  # group_idx * rank
            full_end_row = (group_idx + 1) * (full_shape[0] // num_groups)
            full_weight[full_start_row:full_end_row, :] = group_full
        
        return full_weight
    else:
        return None


def verify_weights_match(basic_layer, vanilla_layer, tp_pg, rank: int, tp_size: int, hidden_size: int, intermediate_size: int, attn_rank: int, mlp_rank: int):
    """
    Verify that Vanilla weights (after gathering shards) match Basic weights.
    All ranks must participate in gathers, but only rank 0 performs verification.
    """
    # All ranks participate in gathers (required for all_gather), but only rank 0 verifies
    if rank == 0:
        print("\n" + "=" * 80)
        print("STEP 1: Verifying Initial Weights Match (Basic vs Vanilla)")
        print("=" * 80)
    
    all_match = True
    
    with torch.no_grad():
        # Layer norms (not sharded)
        if rank == 0:
            print("\n--- Layer Norms ---")
            input_ln_match = torch.allclose(
                basic_layer.input_layernorm.weight,
                vanilla_layer.input_layernorm.weight,
                rtol=1e-5, atol=1e-5
            )
            post_ln_match = torch.allclose(
                basic_layer.post_attention_layernorm.weight,
                vanilla_layer.post_attention_layernorm.weight,
                rtol=1e-5, atol=1e-5
            )
            print(f"  input_layernorm: {'✓' if input_ln_match else '✗'}")
            if not input_ln_match:
                diff = (basic_layer.input_layernorm.weight - vanilla_layer.input_layernorm.weight).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                all_match = False
            print(f"  post_attention_layernorm: {'✓' if post_ln_match else '✗'}")
            if not post_ln_match:
                diff = (basic_layer.post_attention_layernorm.weight - vanilla_layer.post_attention_layernorm.weight).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                all_match = False
        else:
            # Non-rank-0: just check that weights exist (they should match since not sharded)
            input_ln_match = True
            post_ln_match = True
        
        # Attention QKV proj0
        # All ranks participate in gather
        # Use specialized gather for grouped weights [Q;K;V]
        vanilla_qkv_proj0_full = gather_vanilla_grouped_weight_shard(
            vanilla_layer.attn.qkv_proj0.weight,
            (3 * attn_rank, hidden_size),
            num_groups=3,  # Q, K, V
            tp_pg=tp_pg, rank=rank, tp_size=tp_size
        )
        
        if rank == 0:
            print("\n--- Attention QKV proj0 ---")
            # Construct expected from Basic
            q_cola_a_t = basic_layer.attn.q_proj.cola_a.t()  # [rank, hidden_size]
            k_cola_a_t = basic_layer.attn.k_proj.cola_a.t()
            v_cola_a_t = basic_layer.attn.v_proj.cola_a.t()
            basic_qkv_proj0_full = torch.cat([q_cola_a_t, k_cola_a_t, v_cola_a_t], dim=0)  # [3*rank, hidden_size]
            
            print(f"    Basic qkv_proj0 shape: {basic_qkv_proj0_full.shape}")
            print(f"    Vanilla qkv_proj0 shape (after gather): {vanilla_qkv_proj0_full.shape}")
            
            # Check individual Q, K, V slices
            rank_size = attn_rank
            basic_q_slice = basic_qkv_proj0_full[0:rank_size, :]  # [rank, hidden_size]
            basic_k_slice = basic_qkv_proj0_full[rank_size:2*rank_size, :]
            basic_v_slice = basic_qkv_proj0_full[2*rank_size:3*rank_size, :]
            
            vanilla_q_slice = vanilla_qkv_proj0_full[0:rank_size, :]
            vanilla_k_slice = vanilla_qkv_proj0_full[rank_size:2*rank_size, :]
            vanilla_v_slice = vanilla_qkv_proj0_full[2*rank_size:3*rank_size, :]
            
            q_slice_match = torch.allclose(basic_q_slice, vanilla_q_slice, rtol=1e-5, atol=1e-5)
            k_slice_match = torch.allclose(basic_k_slice, vanilla_k_slice, rtol=1e-5, atol=1e-5)
            v_slice_match = torch.allclose(basic_v_slice, vanilla_v_slice, rtol=1e-5, atol=1e-5)
            
            print(f"    Q slice (rows 0:{rank_size}): {'✓' if q_slice_match else '✗'}")
            if not q_slice_match:
                q_diff = (basic_q_slice - vanilla_q_slice).abs()
                print(f"      Q slice - Max diff: {q_diff.max().item():.6e}, Mean diff: {q_diff.mean().item():.6e}")
                print(f"      Basic Q slice sample (first 5 values): {basic_q_slice.flatten()[:5].tolist()}")
                print(f"      Vanilla Q slice sample (first 5 values): {vanilla_q_slice.flatten()[:5].tolist()}")
            
            print(f"    K slice (rows {rank_size}:{2*rank_size}): {'✓' if k_slice_match else '✗'}")
            if not k_slice_match:
                k_diff = (basic_k_slice - vanilla_k_slice).abs()
                print(f"      K slice - Max diff: {k_diff.max().item():.6e}, Mean diff: {k_diff.mean().item():.6e}")
            
            print(f"    V slice (rows {2*rank_size}:{3*rank_size}): {'✓' if v_slice_match else '✗'}")
            if not v_slice_match:
                v_diff = (basic_v_slice - vanilla_v_slice).abs()
                print(f"      V slice - Max diff: {v_diff.max().item():.6e}, Mean diff: {v_diff.mean().item():.6e}")
            
            qkv_proj0_match = torch.allclose(basic_qkv_proj0_full, vanilla_qkv_proj0_full, rtol=1e-5, atol=1e-5)
            print(f"  qkv_proj0 (full): {'✓' if qkv_proj0_match else '✗'}")
            if not qkv_proj0_match:
                diff = (basic_qkv_proj0_full - vanilla_qkv_proj0_full).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                print(f"    Basic sample (first 10 values): {basic_qkv_proj0_full.flatten()[:10].tolist()}")
                print(f"    Vanilla sample (first 10 values): {vanilla_qkv_proj0_full.flatten()[:10].tolist()}")
                all_match = False
            elif not (q_slice_match and k_slice_match and v_slice_match):
                # If full matches but slices don't, something is wrong
                print(f"    WARNING: Full tensor matches but individual slices don't!")
                all_match = False
        else:
            qkv_proj0_match = True
        
        # Q/K/V proj1
        if rank == 0:
            print("\n--- Attention Q/K/V proj1 ---")
        for proj_name in ["q", "k", "v"]:
            vanilla_proj1 = getattr(vanilla_layer.attn, f"{proj_name}_proj1")
            # All ranks participate in gather
            vanilla_proj1_full = gather_vanilla_weight_shard(
                vanilla_proj1.weight,
                (hidden_size, attn_rank),  # [out_features, rank]
                tp_pg, rank, tp_size
            )
            
            if rank == 0:
                basic_proj = getattr(basic_layer.attn, f"{proj_name}_proj")
                # Expected from Basic
                basic_proj1_full = basic_proj.cola_b.t()  # [out_features, rank]
                
                proj1_match = torch.allclose(basic_proj1_full, vanilla_proj1_full, rtol=1e-5, atol=1e-5)
                print(f"  {proj_name}_proj1: {'✓' if proj1_match else '✗'}")
                if not proj1_match:
                    diff = (basic_proj1_full - vanilla_proj1_full).abs()
                    print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                    print(f"    Basic shape: {basic_proj1_full.shape}, Vanilla shape: {vanilla_proj1_full.shape}")
                    all_match = False
        
        # O proj0
        vanilla_o_proj0_full = gather_vanilla_weight_shard(
            vanilla_layer.attn.o_proj0.weight,
            (attn_rank, hidden_size),
            tp_pg, rank, tp_size
        )
        if rank == 0:
            print("\n--- Attention O proj0 ---")
            basic_o_proj0_full = basic_layer.attn.o_proj.cola_a.t()  # [rank, hidden_size]
            o_proj0_match = torch.allclose(basic_o_proj0_full, vanilla_o_proj0_full, rtol=1e-5, atol=1e-5)
            print(f"  o_proj0: {'✓' if o_proj0_match else '✗'}")
            if not o_proj0_match:
                diff = (basic_o_proj0_full - vanilla_o_proj0_full).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                all_match = False
        else:
            o_proj0_match = True
        
        # O proj1
        vanilla_o_proj1_full = gather_vanilla_weight_shard(
            vanilla_layer.attn.o_proj1.weight,
            (hidden_size, attn_rank),
            tp_pg, rank, tp_size
        )
        if rank == 0:
            print("\n--- Attention O proj1 ---")
            basic_o_proj1_full = basic_layer.attn.o_proj.cola_b.t()  # [hidden_size, rank]
            o_proj1_match = torch.allclose(basic_o_proj1_full, vanilla_o_proj1_full, rtol=1e-5, atol=1e-5)
            print(f"  o_proj1: {'✓' if o_proj1_match else '✗'}")
            if not o_proj1_match:
                diff = (basic_o_proj1_full - vanilla_o_proj1_full).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                all_match = False
        else:
            o_proj1_match = True
        
        # MLP gate_up_proj0
        # Use specialized gather for grouped weights [gate;up]
        vanilla_gate_up_proj0_full = gather_vanilla_grouped_weight_shard(
            vanilla_layer.mlp.gate_up_proj0.weight,
            (2 * mlp_rank, hidden_size),
            num_groups=2,  # gate, up
            tp_pg=tp_pg, rank=rank, tp_size=tp_size
        )
        if rank == 0:
            print("\n--- MLP gate_up_proj0 ---")
            gate_cola_a_t = basic_layer.mlp.gate_proj.cola_a.t()
            up_cola_a_t = basic_layer.mlp.up_proj.cola_a.t()
            basic_gate_up_proj0_full = torch.cat([gate_cola_a_t, up_cola_a_t], dim=0)
            gate_up_proj0_match = torch.allclose(basic_gate_up_proj0_full, vanilla_gate_up_proj0_full, rtol=1e-5, atol=1e-5)
            print(f"  gate_up_proj0: {'✓' if gate_up_proj0_match else '✗'}")
            if not gate_up_proj0_match:
                diff = (basic_gate_up_proj0_full - vanilla_gate_up_proj0_full).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                all_match = False
        else:
            gate_up_proj0_match = True
        
        # Gate/Up proj1
        if rank == 0:
            print("\n--- MLP Gate/Up proj1 ---")
        for proj_name in ["gate", "up"]:
            vanilla_proj1 = getattr(vanilla_layer.mlp, f"{proj_name}_proj1")
            vanilla_proj1_full = gather_vanilla_weight_shard(
                vanilla_proj1.weight,
                (intermediate_size, mlp_rank),
                tp_pg, rank, tp_size
            )
            if rank == 0:
                basic_proj = getattr(basic_layer.mlp, f"{proj_name}_proj")
                basic_proj1_full = basic_proj.cola_b.t()  # [intermediate_size, rank]
                proj1_match = torch.allclose(basic_proj1_full, vanilla_proj1_full, rtol=1e-5, atol=1e-5)
                print(f"  {proj_name}_proj1: {'✓' if proj1_match else '✗'}")
                if not proj1_match:
                    diff = (basic_proj1_full - vanilla_proj1_full).abs()
                    print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                    all_match = False
        
        # Down proj0
        vanilla_down_proj0_full = gather_vanilla_weight_shard(
            vanilla_layer.mlp.down_proj0.weight,
            (mlp_rank, intermediate_size),
            tp_pg, rank, tp_size
        )
        if rank == 0:
            print("\n--- MLP down_proj0 ---")
            basic_down_proj0_full = basic_layer.mlp.down_proj.cola_a.t()  # [rank, intermediate_size]
            down_proj0_match = torch.allclose(basic_down_proj0_full, vanilla_down_proj0_full, rtol=1e-5, atol=1e-5)
            print(f"  down_proj0: {'✓' if down_proj0_match else '✗'}")
            if not down_proj0_match:
                diff = (basic_down_proj0_full - vanilla_down_proj0_full).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                all_match = False
        else:
            down_proj0_match = True
        
        # Down proj1
        vanilla_down_proj1_full = gather_vanilla_weight_shard(
            vanilla_layer.mlp.down_proj1.weight,
            (hidden_size, mlp_rank),
            tp_pg, rank, tp_size
        )
        if rank == 0:
            print("\n--- MLP down_proj1 ---")
            basic_down_proj1_full = basic_layer.mlp.down_proj.cola_b.t()  # [hidden_size, rank]
            down_proj1_match = torch.allclose(basic_down_proj1_full, vanilla_down_proj1_full, rtol=1e-5, atol=1e-5)
            print(f"  down_proj1: {'✓' if down_proj1_match else '✗'}")
            if not down_proj1_match:
                diff = (basic_down_proj1_full - vanilla_down_proj1_full).abs()
                print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                all_match = False
        else:
            down_proj1_match = True
    
    if rank == 0:
        print("\n" + "=" * 80)
        if all_match:
            print("✓ All weights match!")
        else:
            print("✗ Some weights do NOT match!")
        print("=" * 80 + "\n")
    
    return all_match if rank == 0 else True


def gather_if_sharded(tensor: torch.Tensor, full_size: int, tp_pg, rank: int, tp_size: int) -> torch.Tensor:
    """
    Gather tensor if it's sharded, otherwise return as-is.
    
    Args:
        tensor: Tensor that may be sharded [..., size_local] or [..., full_size]
        full_size: Expected full size in the last dimension
        tp_pg: Tensor parallel process group
        rank: Current rank
        tp_size: Tensor parallel size
    
    Returns:
        Full tensor on rank 0, None on other ranks (or original tensor if not sharded)
    """
    if tensor.shape[-1] < full_size:
        # Sharded, need to gather
        shard_list = [torch.empty_like(tensor) for _ in range(tp_size)]
        dist.all_gather(shard_list, tensor.contiguous(), group=tp_pg)
        if rank == 0:
            return torch.cat(shard_list, dim=-1)
        else:
            return None
    else:
        # Already full
        if rank == 0:
            return tensor
        else:
            return None


def compare_step_by_step(basic_layer, vanilla_layer, hidden_states, sequence_mask, layer_idx: int, tp_group, rank: int, tp_size: int, hidden_size: int, intermediate_size: int, attn_rank: int, mlp_rank: int, dtype: torch.dtype, num_hidden_layers: int):
    """
    Step-by-step comparison of intermediate outputs between Basic and Vanilla.
    All ranks participate in gathers, but only rank 0 does the comparison.
    """
    # All ranks need to participate in vanilla forward passes for gathers to work
    # Only rank 0 will do basic_layer computations and comparisons
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("STEP 2: Step-by-Step Output Comparison")
        print("=" * 80)
    
    all_steps_match = True if rank == 0 else None
    
    with torch.no_grad():
        # Step 2.1: Input Layer Norm
        if rank == 0:
            print("\n--- Step 2.1: Input Layer Norm ---")
        basic_ln_out = basic_layer.input_layernorm(hidden_states) if rank == 0 and basic_layer is not None else None
        vanilla_ln_out = vanilla_layer.input_layernorm(hidden_states)  # All ranks compute this
        vanilla_ln_out_full = gather_if_sharded(vanilla_ln_out, hidden_size, tp_group, rank, tp_size)
        
        if rank == 0:
            if basic_ln_out is not None:
                print(f"  basic_ln_out shape: {basic_ln_out.shape}")
            print(f"  vanilla_ln_out shape (local, before gather): {vanilla_ln_out.shape}")
            if vanilla_ln_out_full is not None:
                print(f"  vanilla_ln_out_full shape (after gather): {vanilla_ln_out_full.shape}")
        
        if rank == 0 and basic_ln_out is not None and vanilla_ln_out_full is not None:
            ln_match = torch.allclose(basic_ln_out, vanilla_ln_out_full, rtol=1e-3, atol=1e-3)
            diff = (basic_ln_out - vanilla_ln_out_full).abs()
            print(f"  Input LN: {'✓' if ln_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not ln_match:
                all_steps_match = False
        elif rank == 0:
            print("  Input LN: Could not gather Vanilla output")
            all_steps_match = False
        
        # Step 2.2: Attention QKV Projections
        if rank == 0:
            print("\n--- Step 2.2: Attention QKV Projections ---")
        
        # Basic: manual computation (only on rank 0)
        if rank == 0 and basic_layer is not None and basic_ln_out is not None:
            basic_q_lr = torch.matmul(basic_ln_out, basic_layer.attn.q_proj.cola_a)  # [seq, batch, rank]
            basic_q_lr_act = basic_layer.attn.q_proj.lr_act(basic_q_lr)
            basic_q = torch.matmul(basic_q_lr_act, basic_layer.attn.q_proj.cola_b)  # [seq, batch, out]
            
            basic_k_lr = torch.matmul(basic_ln_out, basic_layer.attn.k_proj.cola_a)
            basic_k_lr_act = basic_layer.attn.k_proj.lr_act(basic_k_lr)
            basic_k = torch.matmul(basic_k_lr_act, basic_layer.attn.k_proj.cola_b)
            
            basic_v_lr = torch.matmul(basic_ln_out, basic_layer.attn.v_proj.cola_a)
            basic_v_lr_act = basic_layer.attn.v_proj.lr_act(basic_v_lr)
            basic_v = torch.matmul(basic_v_lr_act, basic_layer.attn.v_proj.cola_b)
        else:
            basic_q_lr_act = None
            basic_q = None
            basic_k = None
            basic_v = None
        
        # Vanilla: call actual forward passes (all ranks)
        vanilla_qkv_lr = vanilla_layer.attn.qkv_proj0(vanilla_ln_out)  # [seq, batch, 3*rank//tp]
        vanilla_qkv_lr_act = vanilla_layer.attn.lr_act(vanilla_qkv_lr)
        
        # Split into q, k, v
        lr_size = vanilla_qkv_lr_act.shape[-1] // 3
        vanilla_lr_q = vanilla_qkv_lr_act[:, :, 0:lr_size]
        vanilla_lr_k = vanilla_qkv_lr_act[:, :, lr_size:2*lr_size]
        vanilla_lr_v = vanilla_qkv_lr_act[:, :, 2*lr_size:3*lr_size]
        
        vanilla_q = vanilla_layer.attn.q_proj1(vanilla_lr_q)
        vanilla_k = vanilla_layer.attn.k_proj1(vanilla_lr_k)
        vanilla_v = vanilla_layer.attn.v_proj1(vanilla_lr_v)
        
        # Gather Vanilla outputs if sharded (all ranks participate)
        vanilla_q_full = gather_if_sharded(vanilla_q, hidden_size, tp_group, rank, tp_size)
        vanilla_k_full = gather_if_sharded(vanilla_k, hidden_size, tp_group, rank, tp_size)
        vanilla_v_full = gather_if_sharded(vanilla_v, hidden_size, tp_group, rank, tp_size)
        
        # Compare low-rank activations (gather if needed, all ranks participate)
        vanilla_lr_q_full = gather_if_sharded(vanilla_lr_q, attn_rank, tp_group, rank, tp_size)
        vanilla_lr_k_full = gather_if_sharded(vanilla_lr_k, attn_rank, tp_group, rank, tp_size)
        vanilla_lr_v_full = gather_if_sharded(vanilla_lr_v, attn_rank, tp_group, rank, tp_size)
        
        if rank == 0:
            if basic_q_lr_act is not None:
                print(f"  basic_q_lr_act shape: {basic_q_lr_act.shape}")
            print(f"  vanilla_lr_q shape (local): {vanilla_lr_q.shape}")
            if vanilla_lr_q_full is not None:
                print(f"  vanilla_lr_q_full shape (after gather): {vanilla_lr_q_full.shape}")
            if basic_q is not None:
                print(f"  basic_q shape: {basic_q.shape}")
            print(f"  vanilla_q shape (local): {vanilla_q.shape}")
            if vanilla_q_full is not None:
                print(f"  vanilla_q_full shape (after gather): {vanilla_q_full.shape}")
            if basic_k is not None:
                print(f"  basic_k shape: {basic_k.shape}")
            print(f"  vanilla_k shape (local): {vanilla_k.shape}")
            if vanilla_k_full is not None:
                print(f"  vanilla_k_full shape (after gather): {vanilla_k_full.shape}")
            if basic_v is not None:
                print(f"  basic_v shape: {basic_v.shape}")
            print(f"  vanilla_v shape (local): {vanilla_v.shape}")
            if vanilla_v_full is not None:
                print(f"  vanilla_v_full shape (after gather): {vanilla_v_full.shape}")
        
        # Compare (only on rank 0)
        if rank == 0 and basic_q_lr_act is not None and vanilla_lr_q_full is not None:
            q_lr_match = torch.allclose(basic_q_lr_act, vanilla_lr_q_full, rtol=1e-3, atol=1e-3)
            diff = (basic_q_lr_act - vanilla_lr_q_full).abs()
            print(f"  Q LR act: {'✓' if q_lr_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not q_lr_match:
                all_steps_match = False
        
        if rank == 0 and basic_q is not None and vanilla_q_full is not None:
            q_match = torch.allclose(basic_q, vanilla_q_full, rtol=1e-3, atol=1e-3)
            diff = (basic_q - vanilla_q_full).abs()
            print(f"  Q final: {'✓' if q_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not q_match:
                all_steps_match = False
                
                # Check 1: Actually recompute in fp32 (not just cast) - only if current dtype is bf16
                if dtype == torch.bfloat16:
                    print(f"\n  --- Check 1: FP32 Recomputation ---")
                    if basic_ln_out is not None and basic_layer is not None:
                        # Recompute Basic Q projection in fp32
                        basic_ln_out_fp32 = basic_ln_out.to(torch.float32)
                        basic_q_lr_fp32 = torch.matmul(basic_ln_out_fp32, basic_layer.attn.q_proj.cola_a.to(torch.float32))
                        basic_q_lr_act_fp32 = basic_layer.attn.q_proj.lr_act(basic_q_lr_fp32.to(torch.bfloat16)).to(torch.float32)  # lr_act might be in bf16
                        basic_q_fp32 = torch.matmul(basic_q_lr_act_fp32, basic_layer.attn.q_proj.cola_b.to(torch.float32))
                        
                        # Recompute Vanilla Q projection in fp32 (need to simulate TP)
                        vanilla_ln_out_fp32 = vanilla_ln_out_full.to(torch.float32) if vanilla_ln_out_full is not None else None
                        if vanilla_ln_out_fp32 is not None:
                            # Simulate vanilla forward in fp32
                            rank_per_tp = attn_rank // tp_size
                            # Get the exact cola_b shards that were loaded into each TP rank
                            cola_b_full_fp32 = basic_layer.attn.q_proj.cola_b.to(torch.float32)  # [64, 128]
                            # Simulate: each rank computes with its shard, then all-reduce
                            partials_fp32 = []
                            for r in range(tp_size):
                                # Each rank gets lr_act shard (we need to simulate this from full)
                                # Actually, we should use the same lr_act as basic, but split it
                                lr_act_shard = basic_q_lr_act_fp32[:, :, r * rank_per_tp:(r + 1) * rank_per_tp]  # [seq, batch, 16]
                                cola_b_shard = cola_b_full_fp32[r * rank_per_tp:(r + 1) * rank_per_tp, :]  # [16, 128]
                                partial = torch.matmul(lr_act_shard, cola_b_shard)  # [seq, batch, 128]
                                partials_fp32.append(partial)
                            vanilla_q_sim_fp32 = sum(partials_fp32)  # All-reduce (sum) in fp32
                            
                            q_match_fp32 = torch.allclose(basic_q_fp32, vanilla_q_sim_fp32, rtol=1e-5, atol=1e-5)
                            diff_fp32 = (basic_q_fp32 - vanilla_q_sim_fp32).abs()
                            print(f"    Q final (fp32 recompute): {'✓' if q_match_fp32 else '✗'} - Max diff: {diff_fp32.max().item():.6e}, Mean diff: {diff_fp32.mean().item():.6e}")
                            if q_match_fp32:
                                print(f"    → Mismatch is purely numeric (bf16 precision)")
                else:
                    print(f"\n  --- Check 1: FP32 Recomputation ---")
                    print(f"    Skipped (already running in fp32)")
                
                # Check 2: Exactly replicate vanilla's kernel path (same dtype GEMM per shard + same dtype all-reduce)
                print(f"\n  --- Check 2: Exact TP Simulation ({dtype} GEMM + {dtype} all-reduce) ---")
                if basic_q_lr_act is not None and basic_layer is not None:
                    rank_per_tp = attn_rank // tp_size  # 64 // 4 = 16
                    cola_b_full = basic_layer.attn.q_proj.cola_b  # [64, 128] in current dtype
                    
                    # Build exact local shards of cola_b (same slices loaded into each TP rank)
                    # These are the exact weights that each TP rank has (from load_vanilla_weights_from_broadcasted_basic)
                    cola_b_shards = []
                    for r in range(tp_size):
                        # This is exactly what each rank r gets: cola_b[r*16:(r+1)*16, :]
                        cola_b_shard = cola_b_full[r * rank_per_tp:(r + 1) * rank_per_tp, :].clone()  # [16, 128]
                        cola_b_shards.append(cola_b_shard)
                        print(f"    Rank {r} cola_b shard shape: {cola_b_shard.shape}, dtype: {cola_b_shard.dtype}")
                    
                    # Recompute partial GEMMs in the same dtype as vanilla kernel
                    # Each rank computes: lr_act_shard @ cola_b_shard
                    partials = []
                    for r in range(tp_size):
                        # Split lr_act into shards (same as TP - each rank gets its portion)
                        lr_act_shard = basic_q_lr_act[:, :, r * rank_per_tp:(r + 1) * rank_per_tp].clone()  # [seq, batch, 16]
                        cola_b_shard = cola_b_shards[r]  # [16, 128]
                        
                        # Ensure both are in the target dtype (like the real kernel)
                        lr_act_shard_dtype = lr_act_shard.to(dtype)
                        cola_b_shard_dtype = cola_b_shard.to(dtype)
                        
                        # Compute GEMM in target dtype: lr_act_shard @ cola_b_shard
                        # Result: [seq, batch, 16] @ [16, 128] = [seq, batch, 128]
                        partial = torch.matmul(lr_act_shard_dtype, cola_b_shard_dtype)  # [seq, batch, 128] in target dtype
                        partials.append(partial)
                        print(f"    Rank {r} partial shape: {partial.shape}, dtype: {partial.dtype}")
                    
                    # Sum partials in target dtype (mimicking all-reduce accumulation)
                    # All-reduce accumulates in the same dtype, so we sum in that dtype
                    basic_q_simulated_tp = partials[0].clone()
                    for p in partials[1:]:
                        basic_q_simulated_tp = (basic_q_simulated_tp.to(dtype) + p.to(dtype)).to(dtype)  # Accumulate in target dtype
                    
                    print(f"    Simulated TP output shape: {basic_q_simulated_tp.shape}, dtype: {basic_q_simulated_tp.dtype}")
                    
                    # Compare with vanilla (convert both to fp32 for comparison if needed)
                    if dtype == torch.float32:
                        basic_q_sim_compare = basic_q_simulated_tp
                        vanilla_q_full_compare = vanilla_q_full
                    else:
                        basic_q_sim_compare = basic_q_simulated_tp.to(torch.float32)
                        vanilla_q_full_compare = vanilla_q_full.to(torch.float32)
                    
                    q_sim_match = torch.allclose(basic_q_sim_compare, vanilla_q_full_compare, rtol=1e-3, atol=1e-3)
                    diff_sim = (basic_q_sim_compare - vanilla_q_full_compare).abs()
                    max_diff = diff_sim.max().item()
                    mean_diff = diff_sim.mean().item()
                    print(f"    Q final (exact TP simulation): {'✓' if q_sim_match else '✗'} - Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
                    
                    if q_sim_match:
                        print(f"    → Implementation is CORRECT! Remaining diff vs Basic is purely 'one GEMM vs four GEMMs + sum' numerical effects")
                    else:
                        # Check bitwise/very close match
                        if max_diff < 1e-2:
                            print(f"    → Very close match (diff < 1e-2), likely correct implementation with small numerical differences")
                            if dtype == torch.bfloat16:
                                print(f"    → Implementation is likely CORRECT, diff is from {dtype} accumulation")
                            else:
                                print(f"    → Implementation is likely CORRECT, diff is from TP accumulation")
                        else:
                            print(f"    → Significant mismatch (diff >= 1e-2), may indicate sharding or computation issue")
                            # Print some sample values for debugging
                            print(f"    Sample values (first 5):")
                            print(f"      Simulated: {basic_q_sim_compare.flatten()[:5].tolist()}")
                            print(f"      Vanilla:   {vanilla_q_full_compare.flatten()[:5].tolist()}")
                
                # Check 3: Check vanilla's RowLinear all-reduce dtype
                print(f"\n  --- Check 3: Vanilla RowLinear All-Reduce Dtype ---")
                print(f"    vanilla_q dtype (before gather): {vanilla_q.dtype}")
                print(f"    vanilla_q_full dtype (after gather): {vanilla_q_full.dtype}")
                vanilla_lr_q_dtype = vanilla_lr_q.dtype if vanilla_lr_q is not None else None
                print(f"    vanilla_lr_q dtype: {vanilla_lr_q_dtype}")
                if vanilla_lr_q is not None:
                    q_proj1_weight_dtype = vanilla_layer.attn.q_proj1.weight.dtype
                    print(f"    vanilla q_proj1.weight dtype: {q_proj1_weight_dtype}")
                    print(f"    Expected output dtype from matmul: {torch.result_type(vanilla_lr_q, vanilla_layer.attn.q_proj1.weight)}")
        
        if rank == 0 and basic_k is not None and vanilla_k_full is not None:
            k_match = torch.allclose(basic_k, vanilla_k_full, rtol=1e-3, atol=1e-3)
            diff = (basic_k - vanilla_k_full).abs()
            print(f"  K final: {'✓' if k_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not k_match:
                all_steps_match = False
        
        if rank == 0 and basic_v is not None and vanilla_v_full is not None:
            v_match = torch.allclose(basic_v, vanilla_v_full, rtol=1e-3, atol=1e-3)
            diff = (basic_v - vanilla_v_full).abs()
            print(f"  V final: {'✓' if v_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not v_match:
                all_steps_match = False
        
        # Step 2.3: Attention Computation
        if rank == 0:
            print("\n--- Step 2.3: Attention Computation ---")
        # Use actual forward passes
        if rank == 0 and basic_layer is not None and basic_ln_out is not None:
            basic_attn_out_dict = basic_layer.attn(hidden_states=basic_ln_out, sequence_mask=sequence_mask)
            basic_attn_out = basic_attn_out_dict["hidden_states"]
        else:
            basic_attn_out = None
        
        vanilla_attn_out_dict = vanilla_layer.attn(hidden_states=vanilla_ln_out, sequence_mask=sequence_mask)  # All ranks
        vanilla_attn_out = vanilla_attn_out_dict["hidden_states"]
        vanilla_attn_out_full = gather_if_sharded(vanilla_attn_out, hidden_size, tp_group, rank, tp_size)
        
        if rank == 0:
            if basic_attn_out is not None:
                print(f"  basic_attn_out shape: {basic_attn_out.shape}")
            print(f"  vanilla_attn_out shape (local): {vanilla_attn_out.shape}")
            if vanilla_attn_out_full is not None:
                print(f"  vanilla_attn_out_full shape (after gather): {vanilla_attn_out_full.shape}")
        
        if rank == 0 and basic_attn_out is not None and vanilla_attn_out_full is not None:
            attn_match = torch.allclose(basic_attn_out, vanilla_attn_out_full, rtol=1e-3, atol=1e-3)
            diff = (basic_attn_out - vanilla_attn_out_full).abs()
            print(f"  Attention output: {'✓' if attn_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not attn_match:
                all_steps_match = False
        
        # Step 2.4: Attention O Projection
        if rank == 0:
            print("\n--- Step 2.4: Attention O Projection ---")
        # Basic: manual computation (only on rank 0)
        if rank == 0 and basic_layer is not None and basic_attn_out is not None:
            basic_o_lr = torch.matmul(basic_attn_out, basic_layer.attn.o_proj.cola_a)
            basic_o_lr_act = basic_layer.attn.o_proj.lr_act(basic_o_lr)
            basic_o_final = torch.matmul(basic_o_lr_act, basic_layer.attn.o_proj.cola_b)
        else:
            basic_o_final = None
        
        # Vanilla: actual forward (all ranks)
        vanilla_o_lr = vanilla_layer.attn.o_proj0(vanilla_attn_out)
        vanilla_o_lr_act = vanilla_layer.attn.lr_act(vanilla_o_lr)
        vanilla_o_final = vanilla_layer.attn.o_proj1(vanilla_o_lr_act)
        vanilla_o_final_full = gather_if_sharded(vanilla_o_final, hidden_size, tp_group, rank, tp_size)
        
        if rank == 0:
            if basic_o_final is not None:
                print(f"  basic_o_final shape: {basic_o_final.shape}")
            print(f"  vanilla_o_final shape (local): {vanilla_o_final.shape}")
            if vanilla_o_final_full is not None:
                print(f"  vanilla_o_final_full shape (after gather): {vanilla_o_final_full.shape}")
        
        if rank == 0 and basic_o_final is not None and vanilla_o_final_full is not None:
            o_match = torch.allclose(basic_o_final, vanilla_o_final_full, rtol=1e-3, atol=1e-3)
            diff = (basic_o_final - vanilla_o_final_full).abs()
            print(f"  O final: {'✓' if o_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not o_match:
                all_steps_match = False
        
        # Step 2.5: After Attention Residual
        if rank == 0:
            print("\n--- Step 2.5: After Attention Residual ---")
        if rank == 0 and basic_o_final is not None:
            basic_after_attn = hidden_states + basic_o_final
        else:
            basic_after_attn = None
        
        # Vanilla: use local sharded tensor for forward pass (all ranks)
        vanilla_after_attn = hidden_states + vanilla_o_final  # All ranks use local sharded vanilla_o_final
        vanilla_after_attn_full = gather_if_sharded(vanilla_after_attn, hidden_size, tp_group, rank, tp_size)  # Only for comparison on rank 0
        
        if rank == 0:
            if basic_after_attn is not None:
                print(f"  basic_after_attn shape: {basic_after_attn.shape}")
            print(f"  vanilla_after_attn shape (local): {vanilla_after_attn.shape}")
            if vanilla_after_attn_full is not None:
                print(f"  vanilla_after_attn_full shape (after gather): {vanilla_after_attn_full.shape}")
        
        if rank == 0 and basic_after_attn is not None and vanilla_after_attn_full is not None:
            after_attn_match = torch.allclose(basic_after_attn, vanilla_after_attn_full, rtol=1e-3, atol=1e-3)
            diff = (basic_after_attn - vanilla_after_attn_full).abs()
            print(f"  After attn residual: {'✓' if after_attn_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not after_attn_match:
                all_steps_match = False
        
        # Step 2.6: Post Attention Layer Norm
        if rank == 0:
            print("\n--- Step 2.6: Post Attention Layer Norm ---")
        if rank == 0 and basic_layer is not None and basic_after_attn is not None:
            basic_post_ln = basic_layer.post_attention_layernorm(basic_after_attn)
        else:
            basic_post_ln = None
        
        # Vanilla: use local sharded tensor for forward pass (all ranks)
        vanilla_post_ln = vanilla_layer.post_attention_layernorm(vanilla_after_attn)  # All ranks use local sharded vanilla_after_attn
        vanilla_post_ln_full = gather_if_sharded(vanilla_post_ln, hidden_size, tp_group, rank, tp_size)  # Only for comparison on rank 0
        
        if rank == 0:
            if basic_post_ln is not None:
                print(f"  basic_post_ln shape: {basic_post_ln.shape}")
            print(f"  vanilla_post_ln shape (local): {vanilla_post_ln.shape}")
            if vanilla_post_ln_full is not None:
                print(f"  vanilla_post_ln_full shape (after gather): {vanilla_post_ln_full.shape}")
        
        if rank == 0 and basic_post_ln is not None and vanilla_post_ln_full is not None:
            post_ln_match = torch.allclose(basic_post_ln, vanilla_post_ln_full, rtol=1e-3, atol=1e-3)
            diff = (basic_post_ln - vanilla_post_ln_full).abs()
            print(f"  Post LN: {'✓' if post_ln_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not post_ln_match:
                all_steps_match = False
        
        # Step 2.7: MLP Gate/Up Projections
        if rank == 0:
            print("\n--- Step 2.7: MLP Gate/Up Projections ---")
        # Basic: manual computation (only on rank 0)
        if rank == 0 and basic_layer is not None and basic_post_ln is not None:
            basic_gate_lr = torch.matmul(basic_post_ln, basic_layer.mlp.gate_proj.cola_a)
            basic_gate_lr_act = basic_layer.mlp.gate_proj.lr_act(basic_gate_lr)
            basic_gate = torch.matmul(basic_gate_lr_act, basic_layer.mlp.gate_proj.cola_b)
            
            basic_up_lr = torch.matmul(basic_post_ln, basic_layer.mlp.up_proj.cola_a)
            basic_up_lr_act = basic_layer.mlp.up_proj.lr_act(basic_up_lr)
            basic_up = torch.matmul(basic_up_lr_act, basic_layer.mlp.up_proj.cola_b)
        else:
            basic_gate = None
            basic_up = None
        
        # Vanilla: actual forward (all ranks) - use local sharded tensor
        vanilla_gate_up_lr = vanilla_layer.mlp.gate_up_proj0(vanilla_post_ln)  # All ranks use local sharded vanilla_post_ln
        vanilla_gate_up_lr_act = vanilla_layer.mlp.lr_act(vanilla_gate_up_lr)
        lr_size_mlp = vanilla_gate_up_lr_act.shape[-1] // 2
        vanilla_lr_gate = vanilla_gate_up_lr_act[:, :, 0:lr_size_mlp]
        vanilla_lr_up = vanilla_gate_up_lr_act[:, :, lr_size_mlp:2*lr_size_mlp]
        
        vanilla_gate = vanilla_layer.mlp.gate_proj1(vanilla_lr_gate)
        vanilla_up = vanilla_layer.mlp.up_proj1(vanilla_lr_up)
        
        # Gather only for comparison on rank 0
        vanilla_gate_full = gather_if_sharded(vanilla_gate, intermediate_size, tp_group, rank, tp_size)
        vanilla_up_full = gather_if_sharded(vanilla_up, intermediate_size, tp_group, rank, tp_size)
        
        if rank == 0:
            if basic_gate is not None:
                print(f"  basic_gate shape: {basic_gate.shape}")
            print(f"  vanilla_gate shape (local): {vanilla_gate.shape}")
            if vanilla_gate_full is not None:
                print(f"  vanilla_gate_full shape (after gather): {vanilla_gate_full.shape}")
            if basic_up is not None:
                print(f"  basic_up shape: {basic_up.shape}")
            print(f"  vanilla_up shape (local): {vanilla_up.shape}")
            if vanilla_up_full is not None:
                print(f"  vanilla_up_full shape (after gather): {vanilla_up_full.shape}")
        
        if rank == 0 and basic_gate is not None and vanilla_gate_full is not None:
            gate_match = torch.allclose(basic_gate, vanilla_gate_full, rtol=1e-3, atol=1e-3)
            diff = (basic_gate - vanilla_gate_full).abs()
            print(f"  Gate final: {'✓' if gate_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not gate_match:
                all_steps_match = False
        
        if rank == 0 and basic_up is not None and vanilla_up_full is not None:
            up_match = torch.allclose(basic_up, vanilla_up_full, rtol=1e-3, atol=1e-3)
            diff = (basic_up - vanilla_up_full).abs()
            print(f"  Up final: {'✓' if up_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not up_match:
                all_steps_match = False
        
        # Step 2.8: MLP Intermediate (gate * up)
        if rank == 0:
            print("\n--- Step 2.8: MLP Intermediate (gate * up) ---")
        if rank == 0 and basic_gate is not None and basic_up is not None:
            basic_mlp_intermediate = basic_gate * basic_up
        else:
            basic_mlp_intermediate = None
        
        # Vanilla: use local sharded tensors for forward pass (all ranks)
        vanilla_mlp_intermediate = vanilla_gate * vanilla_up  # All ranks use local sharded tensors
        vanilla_mlp_intermediate_full = gather_if_sharded(vanilla_mlp_intermediate, intermediate_size, tp_group, rank, tp_size)  # Only for comparison on rank 0
        
        if rank == 0:
            if basic_mlp_intermediate is not None:
                print(f"  basic_mlp_intermediate shape: {basic_mlp_intermediate.shape}")
            print(f"  vanilla_mlp_intermediate shape (local): {vanilla_mlp_intermediate.shape}")
            if vanilla_mlp_intermediate_full is not None:
                print(f"  vanilla_mlp_intermediate_full shape (after gather): {vanilla_mlp_intermediate_full.shape}")
        
        if rank == 0 and basic_mlp_intermediate is not None and vanilla_mlp_intermediate_full is not None:
            intermediate_match = torch.allclose(basic_mlp_intermediate, vanilla_mlp_intermediate_full, rtol=1e-3, atol=1e-3)
            diff = (basic_mlp_intermediate - vanilla_mlp_intermediate_full).abs()
            print(f"  MLP intermediate: {'✓' if intermediate_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not intermediate_match:
                all_steps_match = False
        
        # Step 2.9: MLP Down Projection
        if rank == 0:
            print("\n--- Step 2.9: MLP Down Projection ---")
        # Basic: manual computation (only on rank 0)
        if rank == 0 and basic_layer is not None and basic_mlp_intermediate is not None:
            basic_down_lr = torch.matmul(basic_mlp_intermediate, basic_layer.mlp.down_proj.cola_a)
            basic_down_lr_act = basic_layer.mlp.down_proj.lr_act(basic_down_lr)
            basic_down = torch.matmul(basic_down_lr_act, basic_layer.mlp.down_proj.cola_b)
        else:
            basic_down = None
        
        # Vanilla: actual forward (all ranks) - use local sharded tensor
        vanilla_down_lr = vanilla_layer.mlp.down_proj0(vanilla_mlp_intermediate)  # All ranks use local sharded vanilla_mlp_intermediate
        vanilla_down_lr_act = vanilla_layer.mlp.lr_act(vanilla_down_lr)
        vanilla_down = vanilla_layer.mlp.down_proj1(vanilla_down_lr_act)
        vanilla_down_full = gather_if_sharded(vanilla_down, hidden_size, tp_group, rank, tp_size)
        
        if rank == 0:
            if basic_down is not None:
                print(f"  basic_down shape: {basic_down.shape}")
            print(f"  vanilla_down shape (local): {vanilla_down.shape}")
            if vanilla_down_full is not None:
                print(f"  vanilla_down_full shape (after gather): {vanilla_down_full.shape}")
        
        if rank == 0 and basic_down is not None and vanilla_down_full is not None:
            down_match = torch.allclose(basic_down, vanilla_down_full, rtol=1e-3, atol=1e-3)
            diff = (basic_down - vanilla_down_full).abs()
            print(f"  Down final: {'✓' if down_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not down_match:
                all_steps_match = False
        
        # Step 2.10: Final Output (after MLP residual)
        if rank == 0:
            print("\n--- Step 2.10: Final Output (after MLP residual) ---")
        # Note: Last layer may skip residual add
        if rank == 0 and basic_layer is not None:
            if layer_idx == num_hidden_layers - 1:
                basic_final = basic_down
            else:
                basic_final = basic_after_attn + basic_down if (basic_after_attn is not None and basic_down is not None) else None
        else:
            basic_final = None
        
        # Vanilla: use local sharded tensors for forward pass (all ranks)
        if layer_idx == num_hidden_layers - 1:
            vanilla_final = vanilla_down  # Last layer: no residual
        else:
            vanilla_final = vanilla_after_attn + vanilla_down  # All ranks use local sharded tensors
        vanilla_final_full = gather_if_sharded(vanilla_final, hidden_size, tp_group, rank, tp_size)  # Only for comparison on rank 0
        
        if rank == 0:
            if basic_final is not None:
                print(f"  basic_final shape: {basic_final.shape}")
            print(f"  vanilla_final shape (local): {vanilla_final.shape}")
            if vanilla_final_full is not None:
                print(f"  vanilla_final_full shape (after gather): {vanilla_final_full.shape}")
        
        if rank == 0 and basic_final is not None and vanilla_final_full is not None:
            final_match = torch.allclose(basic_final, vanilla_final_full, rtol=1e-3, atol=1e-3)
            diff = (basic_final - vanilla_final_full).abs()
            print(f"  Final output: {'✓' if final_match else '✗'} - Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if not final_match:
                all_steps_match = False
    
    print("\n" + "=" * 80)
    if all_steps_match:
        print("✓ All steps match!")
    else:
        print("✗ Some steps do NOT match!")
    print("=" * 80 + "\n")
    
    return all_steps_match


def test_layer_parity(layer_idx: int, tp_size: int, vanilla_parallel_context: ParallelContext, dtype: torch.dtype):
    """
    Test parity between Basic (TP=1) and Vanilla (TP=4) for a given layer index.
    
    Args:
        layer_idx: Layer index to test (0 or last layer)
        tp_size: Tensor parallel size (should be 4)
        vanilla_parallel_context: Pre-created ParallelContext (reused across tests)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size != tp_size:
        if rank == 0:
            print(f"ERROR: World size {world_size} != TP size {tp_size}")
        return False
    
    # Use WORLD as TP group (all 4 ranks)
    tp_group = dist.group.WORLD
    
    # Set deterministic seeds (same on all ranks)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create configs
    basic_config, vanilla_config = create_test_configs()
    
    vanilla_parallel_config = ParallelismArgs(
        dp=1,
        tp=tp_size,
        pp=1,
        expert_parallel_size=1,
        recompute_layer=False,
    )
    
    basic_parallel_config = ParallelismArgs(
        dp=1,
        tp=1,
        pp=1,
        expert_parallel_size=1,
        recompute_layer=False,
    )
    
    # Step 1: Build Basic layer on rank 0 only
    basic_layer = None
    if rank == 0:
        basic_layer = BasicColaLlamaDecoderLayer(
            config=basic_config,
            parallel_config=basic_parallel_config,
            tp_pg=tp_group,
            layer_idx=layer_idx,
        ).cuda().eval().to(dtype=dtype)
    
    # Step 2: Extract and broadcast Basic weights (full tensors)
    # All ranks pre-allocate tensors with correct shapes
    hidden_size = basic_config.hidden_size
    intermediate_size = basic_config.intermediate_size
    attn_rank = basic_config.attn_rank
    mlp_rank = basic_config.mlp_rank
    device = "cuda"
    
    # Layer norm weights
    if rank == 0:
        input_ln_weight = basic_layer.input_layernorm.weight.clone()
        post_ln_weight = basic_layer.post_attention_layernorm.weight.clone()
    else:
        input_ln_weight = torch.empty(hidden_size, dtype=dtype, device=device)
        post_ln_weight = torch.empty(hidden_size, dtype=dtype, device=device)
    broadcast_tensor(input_ln_weight, src_rank=0, group=tp_group, rank=rank)
    broadcast_tensor(post_ln_weight, src_rank=0, group=tp_group, rank=rank)
    
    # Attention weights
    if rank == 0:
        q_cola_a = basic_layer.attn.q_proj.cola_a.clone()  # [hidden_size, rank]
        k_cola_a = basic_layer.attn.k_proj.cola_a.clone()
        v_cola_a = basic_layer.attn.v_proj.cola_a.clone()
        q_cola_b = basic_layer.attn.q_proj.cola_b.clone()  # [rank, out_features]
        k_cola_b = basic_layer.attn.k_proj.cola_b.clone()
        v_cola_b = basic_layer.attn.v_proj.cola_b.clone()
        o_cola_a = basic_layer.attn.o_proj.cola_a.clone()  # [hidden_size, rank]
        o_cola_b = basic_layer.attn.o_proj.cola_b.clone()  # [rank, hidden_size]
    else:
        q_cola_a = torch.empty((hidden_size, attn_rank), dtype=dtype, device=device)
        k_cola_a = torch.empty((hidden_size, attn_rank), dtype=dtype, device=device)
        v_cola_a = torch.empty((hidden_size, attn_rank), dtype=dtype, device=device)
        q_cola_b = torch.empty((attn_rank, hidden_size), dtype=dtype, device=device)
        k_cola_b = torch.empty((attn_rank, hidden_size), dtype=dtype, device=device)
        v_cola_b = torch.empty((attn_rank, hidden_size), dtype=dtype, device=device)
        o_cola_a = torch.empty((hidden_size, attn_rank), dtype=dtype, device=device)
        o_cola_b = torch.empty((attn_rank, hidden_size), dtype=dtype, device=device)
    
    # Broadcast attention weights
    for tensor in [q_cola_a, k_cola_a, v_cola_a, q_cola_b, k_cola_b, v_cola_b, o_cola_a, o_cola_b]:
        broadcast_tensor(tensor, src_rank=0, group=tp_group, rank=rank)
    
    # MLP weights
    if rank == 0:
        gate_cola_a = basic_layer.mlp.gate_proj.cola_a.clone()  # [hidden_size, rank]
        up_cola_a = basic_layer.mlp.up_proj.cola_a.clone()
        gate_cola_b = basic_layer.mlp.gate_proj.cola_b.clone()  # [rank, intermediate_size]
        up_cola_b = basic_layer.mlp.up_proj.cola_b.clone()
        down_cola_a = basic_layer.mlp.down_proj.cola_a.clone()  # [intermediate_size, rank]
        down_cola_b = basic_layer.mlp.down_proj.cola_b.clone()  # [rank, hidden_size]
    else:
        gate_cola_a = torch.empty((hidden_size, mlp_rank), dtype=dtype, device=device)
        up_cola_a = torch.empty((hidden_size, mlp_rank), dtype=dtype, device=device)
        gate_cola_b = torch.empty((mlp_rank, intermediate_size), dtype=dtype, device=device)
        up_cola_b = torch.empty((mlp_rank, intermediate_size), dtype=dtype, device=device)
        down_cola_a = torch.empty((intermediate_size, mlp_rank), dtype=dtype, device=device)
        down_cola_b = torch.empty((mlp_rank, hidden_size), dtype=dtype, device=device)
    
    # Broadcast MLP weights
    for tensor in [gate_cola_a, up_cola_a, gate_cola_b, up_cola_b, down_cola_a, down_cola_b]:
        broadcast_tensor(tensor, src_rank=0, group=tp_group, rank=rank)
    
    # Step 3: Build Vanilla TP=4 layer on every rank
    vanilla_layer = VanillaLlamaDecoderLayer(
        config=vanilla_config,
        parallel_config=vanilla_parallel_config,
        tp_pg=vanilla_parallel_context.tp_pg,
        layer_idx=layer_idx,
    ).cuda().eval().to(dtype=dtype)
    
    # Step 4: Load Vanilla weights from broadcasted Basic tensors
    # Each rank extracts its local shard
    load_vanilla_weights_from_broadcasted_basic(
        vanilla_layer,
        input_ln_weight,
        post_ln_weight,
        q_cola_a, k_cola_a, v_cola_a,
        q_cola_b, k_cola_b, v_cola_b,
        o_cola_a, o_cola_b,
        gate_cola_a, up_cola_a,
        gate_cola_b, up_cola_b,
        down_cola_a, down_cola_b,
        rank=rank,
        tp_size=tp_size,
    )
    
    # Step 4.5: Verify weights match (all ranks participate, but only rank 0 verifies)
    weights_match = verify_weights_match(
        basic_layer,
        vanilla_layer,
        tp_group,
        rank,
        tp_size,
        hidden_size,
        intermediate_size,
        attn_rank,
        mlp_rank
    )
    if rank == 0 and not weights_match:
        print("⚠️  WARNING: Weights do not match! Forward pass comparison may be invalid.")
    
    # Step 5: Input broadcast
    seq_len = 32
    batch_size = 2
    input_dtype = dtype  # Use the same dtype as model parameters
    
    if rank == 0:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        hidden_states = torch.randn(
            seq_len, batch_size, hidden_size,
            dtype=input_dtype,
            device=device
        )
        sequence_mask = torch.ones(
            batch_size, seq_len,
            dtype=torch.bool,
            device=device
        )
    else:
        hidden_states = torch.empty(
            seq_len, batch_size, hidden_size,
            dtype=input_dtype,
            device=device
        )
        sequence_mask = torch.empty(
            batch_size, seq_len,
            dtype=torch.bool,
            device=device
        )
    
    # Broadcast inputs
    dist.broadcast(hidden_states, src=0, group=tp_group)
    dist.broadcast(sequence_mask, src=0, group=tp_group)
    
    # Step 6: Forward + gather output for compare
    with torch.no_grad():
        # Vanilla forward (on all ranks)
        vanilla_output = vanilla_layer(
            hidden_states=hidden_states,
            sequence_mask=sequence_mask
        )
        vanilla_hidden = vanilla_output["hidden_states"]  # [seq, batch, hidden_size_local or hidden_size]
        
        # Check if Vanilla output is sharded
        if vanilla_hidden.shape[-1] < hidden_size:
            # Output is sharded, gather shards
            # All ranks must participate in all_gather
            shard_list = [torch.empty_like(vanilla_hidden) for _ in range(tp_size)]
            dist.all_gather(shard_list, vanilla_hidden.contiguous(), group=tp_group)
            # Concatenate along last dimension to reconstruct full tensor
            vanilla_hidden_full = torch.cat(shard_list, dim=-1)  # [seq, batch, hidden_size]
            # Only rank 0 will use it for comparison
            if rank != 0:
                vanilla_hidden_full = None
        else:
            # Output is already full (all-reduced), just use rank 0's copy
            if rank == 0:
                print(f"Output is already full")
                vanilla_hidden_full = vanilla_hidden
            else:
                vanilla_hidden_full = None
        
        # Basic forward (only on rank 0)
        basic_hidden = None
        if rank == 0:
            basic_output = basic_layer(
                hidden_states=hidden_states,
                sequence_mask=sequence_mask
            )
            basic_hidden = basic_output["hidden_states"]
        
        # Step 2: Step-by-step comparison (all ranks participate in gathers, but only rank 0 compares)
        compare_step_by_step(
            basic_layer if rank == 0 else None,
            vanilla_layer,
            hidden_states,
            sequence_mask,
            layer_idx,
            tp_group,
            rank,
            tp_size,
            hidden_size,
            intermediate_size,
            attn_rank,
            mlp_rank,
            dtype,
            basic_config.num_hidden_layers
        )
        
        # Compare on rank 0
        if rank == 0:
            assert basic_hidden is not None, "Basic output should be computed on rank 0"
            assert vanilla_hidden_full is not None, "Vanilla output should be gathered on rank 0"
            
            # Check shapes match
            assert basic_hidden.shape == vanilla_hidden_full.shape, \
                f"Shape mismatch: Basic {basic_hidden.shape} vs Vanilla {vanilla_hidden_full.shape}"
            
            # Compare with tolerance
            match = torch.allclose(basic_hidden, vanilla_hidden_full, rtol=1e-3, atol=1e-3)
            
            diff = (basic_hidden - vanilla_hidden_full).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            layer_name = f"layer_{layer_idx}"
            if match:
                print(f"✓ {layer_name}: Basic (TP=1) matches Vanilla (TP=4)")
                print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
                return True
            else:
                print(f"✗ {layer_name}: Basic (TP=1) does NOT match Vanilla (TP=4)")
                print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
                return False
        else:
            # Other ranks just return True (comparison only happens on rank 0)
            return True


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
    """Main test function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test BasicCoLA (TP=1) vs VanillaCoLA (TP=4) parity")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
        help="Data type to use: 'fp32' or 'bf16' (default: fp32)"
    )
    args = parser.parse_args()
    
    # Map dtype string to torch dtype
    if args.dtype == "fp32":
        dtype = torch.float32
        # Disable TF32 for true FP32 precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        # Patch flash_attn to handle fp32 (it only supports fp16/bf16)
        patched = patch_flash_attn_for_fp32()
        if not patched:
            print("WARNING: Could not patch flash_attn for fp32. FlashAttention only supports fp16/bf16.")
            print("         Consider using --dtype bf16 instead.")
            return
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 80)
        print("Testing BasicCoLA (TP=1) vs VanillaCoLA (TP=4) Parity")
        print("Weight-loading strategy: Broadcast full Basic weights, then shard per rank")
        print("=" * 80)
        print(f"World size: {world_size}, TP size: 4")
        print(f"Data type: {args.dtype} ({dtype})")
        if args.dtype == "fp32":
            print(f"TF32 disabled: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")
            print(f"NOTE: FlashAttention only supports fp16/bf16. FP32 inputs will be converted to bf16")
            print(f"      for flash_attn calls, then converted back to fp32. Other operations remain in fp32.")
        print()
    
    # Create TP4 parallel context once (reused for both tests)
    vanilla_parallel_context = ParallelContext(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    
    try:
        # Test layer 0
        if rank == 0:
            print("Testing layer_idx=0 (first layer)...")
        success_0 = test_layer_parity(layer_idx=0, tp_size=4, vanilla_parallel_context=vanilla_parallel_context, dtype=dtype)
        
        # Test last layer
        if rank == 0:
            print("\nTesting layer_idx=1 (last layer, num_hidden_layers=2)...")
        success_last = test_layer_parity(layer_idx=1, tp_size=4, vanilla_parallel_context=vanilla_parallel_context, dtype=dtype)
        
        # Final summary
        if rank == 0:
            print("\n" + "=" * 80)
            if success_0 and success_last:
                print("✓ All tests passed!")
            else:
                print("✗ Some tests failed!")
            print("=" * 80)
    finally:
        # Clean up parallel context
        vanilla_parallel_context.destroy()
    
    # Clean up process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

