"""
Test to verify forward compute parity between BasicCoLA (TP=1) and VanillaCoLA (TP=4).

This test:
1. Instantiates BasicCoLA decoder layer with TP=1 on rank 0 only (ground truth)
2. Instantiates VanillaCoLA decoder layer with TP=4 on all 4 ranks
3. Copies weights from Basic to Vanilla on rank 0, then broadcasts to all ranks
4. Runs forward pass with identical inputs (broadcast from rank 0)
5. Gathers Vanilla output if needed and compares with Basic output on rank 0

Run with: torchrun --nproc_per_node=4 test_vanilla_tp4_vs_basic_tp1.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

# Add examples/cola to path
cola_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "cola")
sys.path.insert(0, cola_dir)

from config_basic_cola_llama import BasicColaLlamaConfig
from config_cola_llama import ColaLlamaConfig
from basic_cola_llama import BasicColaLlamaDecoderLayer
from vanilla_cola_llama import LlamaDecoderLayer as VanillaLlamaDecoderLayer

from nanotron.config import ParallelismArgs
from nanotron import distributed as dist_nano
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


def copy_basic_cola_to_vanilla_linear(basic_cola_layer, proj0_linear: nn.Module, proj1_linear: nn.Module):
    """
    Copy Basic ColaLayer factors to Vanilla proj0 and proj1.
    
    Basic: cola_a [in, rank], cola_b [rank, out]
    
    For Vanilla:
    - proj0 is ColumnLinear: weight [out_features, in_features] = [rank, hidden_size]
      Need: cola_a.T = [rank, in] ✓
    - proj1 is RowLinear: weight [out_features, in_features] = [out_features, rank]
      Need: cola_b.T = [out, rank] ✓
    """
    with torch.no_grad():
        # Copy cola_a.T to proj0.weight
        # basic.cola_a is [in, rank], transpose to [rank, in]
        proj0_linear.weight.copy_(basic_cola_layer.cola_a.t())
        
        # Copy cola_b.T to proj1.weight
        # basic.cola_b is [rank, out], transpose to [out, rank]
        proj1_linear.weight.copy_(basic_cola_layer.cola_b.t())


def copy_basic_attn_to_vanilla(basic_attn, vanilla_attn):
    """
    Copy Basic attention weights to Vanilla attention.
    
    Vanilla structure:
    - qkv_proj0: ColumnLinear(hidden_size, 3*rank) -> weight [3*rank, hidden_size]
    - q_proj1, k_proj1, v_proj1: RowLinear(rank, out_features) -> weight [out_features, rank]
    - o_proj0: ColumnLinear(hidden_size, rank) -> weight [rank, hidden_size]
    - o_proj1: RowLinear(rank, hidden_size) -> weight [hidden_size, rank]
    """
    with torch.no_grad():
        # QKV projections
        q_cola_a = basic_attn.q_proj.cola_a  # [hidden_size, rank]
        k_cola_a = basic_attn.k_proj.cola_a  # [hidden_size, rank]
        v_cola_a = basic_attn.v_proj.cola_a  # [hidden_size, rank]
        
        # qkv_proj0.weight is [3*rank, hidden_size] (ColumnLinear: [out_features, in_features])
        rank = q_cola_a.shape[1]
        vanilla_attn.qkv_proj0.weight[0:rank, :].copy_(q_cola_a.t())  # [rank, hidden_size]
        vanilla_attn.qkv_proj0.weight[rank:2*rank, :].copy_(k_cola_a.t())
        vanilla_attn.qkv_proj0.weight[2*rank:3*rank, :].copy_(v_cola_a.t())
        
        # Copy qkv_proj1 (cola_b equivalents)
        q_cola_b = basic_attn.q_proj.cola_b  # [rank, out_features]
        k_cola_b = basic_attn.k_proj.cola_b  # [rank, out_features]
        v_cola_b = basic_attn.v_proj.cola_b  # [rank, out_features]
        
        # Vanilla has separate q_proj1, k_proj1, v_proj1 (RowLinear)
        # Each weight is [out_features, rank]
        vanilla_attn.q_proj1.weight.copy_(q_cola_b.t())  # [out_features, rank]
        vanilla_attn.k_proj1.weight.copy_(k_cola_b.t())
        vanilla_attn.v_proj1.weight.copy_(v_cola_b.t())
        
        # O projection
        copy_basic_cola_to_vanilla_linear(
            basic_attn.o_proj,
            vanilla_attn.o_proj0,  # ColumnLinear: [rank, hidden_size]
            vanilla_attn.o_proj1   # RowLinear: [hidden_size, rank]
        )


def copy_basic_mlp_to_vanilla(basic_mlp, vanilla_mlp):
    """
    Copy Basic MLP weights to Vanilla MLP.
    
    Vanilla structure:
    - gate_up_proj0: ColumnLinear(hidden_size, 2*rank) -> weight [2*rank, hidden_size]
    - gate_proj1, up_proj1: RowLinear(rank, intermediate_size) -> weight [intermediate_size, rank]
    - down_proj0: ColumnLinear(intermediate_size, rank) -> weight [rank, intermediate_size]
    - down_proj1: RowLinear(rank, hidden_size) -> weight [hidden_size, rank]
    """
    with torch.no_grad():
        # Gate and Up projections
        gate_cola_a = basic_mlp.gate_proj.cola_a  # [hidden_size, rank]
        up_cola_a = basic_mlp.up_proj.cola_a  # [hidden_size, rank]
        
        rank = gate_cola_a.shape[1]
        # gate_up_proj0.weight is [2*rank, hidden_size] (ColumnLinear)
        vanilla_mlp.gate_up_proj0.weight[0:rank, :].copy_(gate_cola_a.t())  # [rank, hidden_size]
        vanilla_mlp.gate_up_proj0.weight[rank:2*rank, :].copy_(up_cola_a.t())
        
        gate_cola_b = basic_mlp.gate_proj.cola_b  # [rank, intermediate_size]
        up_cola_b = basic_mlp.up_proj.cola_b  # [rank, intermediate_size]
        
        # Vanilla has separate gate_proj1, up_proj1 (RowLinear)
        # Each weight is [intermediate_size, rank]
        vanilla_mlp.gate_proj1.weight.copy_(gate_cola_b.t())  # [intermediate_size, rank]
        vanilla_mlp.up_proj1.weight.copy_(up_cola_b.t())
        
        # Down projection
        copy_basic_cola_to_vanilla_linear(
            basic_mlp.down_proj,
            vanilla_mlp.down_proj0,  # ColumnLinear: [rank, intermediate_size]
            vanilla_mlp.down_proj1   # RowLinear: [hidden_size, rank]
        )


def copy_basic_to_vanilla(basic_layer, vanilla_layer):
    """Copy all weights from Basic layer to Vanilla layer"""
    with torch.no_grad():
        # Copy layer norms
        if hasattr(basic_layer, 'input_layernorm') and hasattr(vanilla_layer, 'input_layernorm'):
            vanilla_layer.input_layernorm.weight.copy_(basic_layer.input_layernorm.weight)
            if hasattr(basic_layer.input_layernorm, 'bias') and basic_layer.input_layernorm.bias is not None:
                vanilla_layer.input_layernorm.bias.copy_(basic_layer.input_layernorm.bias)
        
        if hasattr(basic_layer, 'post_attention_layernorm') and hasattr(vanilla_layer, 'post_attention_layernorm'):
            vanilla_layer.post_attention_layernorm.weight.copy_(basic_layer.post_attention_layernorm.weight)
            if hasattr(basic_layer.post_attention_layernorm, 'bias') and basic_layer.post_attention_layernorm.bias is not None:
                vanilla_layer.post_attention_layernorm.bias.copy_(basic_layer.post_attention_layernorm.bias)
        
        # Copy attention
        copy_basic_attn_to_vanilla(basic_layer.attn, vanilla_layer.attn)
        
        # Copy MLP
        copy_basic_mlp_to_vanilla(basic_layer.mlp, vanilla_layer.mlp)


def distribute_basic_weights_to_vanilla_tp(basic_layer: Optional[nn.Module], vanilla_layer, tp_pg, rank, tp_size):
    """
    Distribute Basic (TP=1) weights to Vanilla (TP=4) across all TP ranks.
    
    Strategy:
    1. On rank 0: Copy Basic weights to get full weight tensors
    2. For each TP-sharded parameter in Vanilla:
       - Get the full weight from Basic
       - Broadcast full weight from rank 0 to all ranks
       - On each rank, extract the appropriate local shard based on module type
       - Set each rank's local shard
    
    TP Sharding Pattern:
    - ColumnLinear: Shards output dimension (first dim of weight matrix)
      Example: weight [out_features, in_features] -> each rank gets [out_features//tp_size, in_features]
    - RowLinear: Shards input dimension (second dim of weight matrix)
      Example: weight [out_features, in_features] -> each rank gets [out_features, in_features//tp_size]
    
    Note: This ensures each TP rank has the correct local shard of the weights.
    """
    from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear, TensorParallelRowLinear
    
    with torch.no_grad():
        # Step 1: On rank 0, prepare full weight tensors from Basic
        if rank == 0:
            # Copy to vanilla on rank 0 first (this sets rank 0's local shards)
            copy_basic_to_vanilla(basic_layer, vanilla_layer)
        
        # Step 2: For each TP parameter, extract and distribute shards
        # We'll iterate through vanilla parameters and extract shards from Basic weights
        
        # Get dtype and device from vanilla parameters
        sample_param = next(vanilla_layer.parameters())
        dtype = sample_param.dtype
        device = sample_param.device
        
        # Attention QKV proj0 (ColumnLinear: shards output dim)
        if rank == 0:
            q_cola_a = basic_layer.attn.q_proj.cola_a.t()  # [rank, hidden_size]
            k_cola_a = basic_layer.attn.k_proj.cola_a.t()  # [rank, hidden_size]
            v_cola_a = basic_layer.attn.v_proj.cola_a.t()  # [rank, hidden_size]
            full_qkv_proj0 = torch.cat([q_cola_a, k_cola_a, v_cola_a], dim=0)  # [3*rank, hidden_size]
        else:
            full_qkv_proj0 = torch.empty((3*64, 128), dtype=dtype, device=device)
        dist.broadcast(full_qkv_proj0, src=0, group=tp_pg)
        
        # Extract shards for each rank (ColumnLinear shards output dim)
        rank_per_shard = full_qkv_proj0.shape[0] // tp_size  # 192 // 4 = 48
        start_idx = rank * rank_per_shard
        end_idx = (rank + 1) * rank_per_shard
        vanilla_layer.attn.qkv_proj0.weight.copy_(full_qkv_proj0[start_idx:end_idx, :])
        
        # Q/K/V proj1 (RowLinear: shards input dim)
        for proj_name in ["q", "k", "v"]:
            if rank == 0:
                basic_proj = getattr(basic_layer.attn, f"{proj_name}_proj")
                full_proj1 = basic_proj.cola_b.t()  # [out_features, rank]
            else:
                full_proj1 = torch.empty((128, 64), dtype=dtype, device=device)
            dist.broadcast(full_proj1, src=0, group=tp_pg)
            
            # RowLinear shards input dim (second dimension)
            rank_per_shard = full_proj1.shape[1] // tp_size  # 64 // 4 = 16
            start_idx = rank * rank_per_shard
            end_idx = (rank + 1) * rank_per_shard
            getattr(vanilla_layer.attn, f"{proj_name}_proj1").weight.copy_(
                full_proj1[:, start_idx:end_idx]
            )
        
        # O proj0 (ColumnLinear: shards output dim)
        if rank == 0:
            full_o_proj0 = basic_layer.attn.o_proj.cola_a.t()  # [rank, hidden_size]
        else:
            full_o_proj0 = torch.empty((64, 128), dtype=dtype, device=device)
        dist.broadcast(full_o_proj0, src=0, group=tp_pg)
        rank_per_shard = full_o_proj0.shape[0] // tp_size  # 64 // 4 = 16
        start_idx = rank * rank_per_shard
        end_idx = (rank + 1) * rank_per_shard
        vanilla_layer.attn.o_proj0.weight.copy_(full_o_proj0[start_idx:end_idx, :])
        
        # O proj1 (RowLinear: shards input dim)
        if rank == 0:
            full_o_proj1 = basic_layer.attn.o_proj.cola_b.t()  # [hidden_size, rank]
        else:
            full_o_proj1 = torch.empty((128, 64), dtype=dtype, device=device)
        dist.broadcast(full_o_proj1, src=0, group=tp_pg)
        rank_per_shard = full_o_proj1.shape[1] // tp_size  # 64 // 4 = 16
        start_idx = rank * rank_per_shard
        end_idx = (rank + 1) * rank_per_shard
        vanilla_layer.attn.o_proj1.weight.copy_(full_o_proj1[:, start_idx:end_idx])
        
        # MLP gate_up_proj0 (ColumnLinear: shards output dim)
        if rank == 0:
            gate_cola_a = basic_layer.mlp.gate_proj.cola_a.t()  # [rank, hidden_size]
            up_cola_a = basic_layer.mlp.up_proj.cola_a.t()  # [rank, hidden_size]
            full_gate_up_proj0 = torch.cat([gate_cola_a, up_cola_a], dim=0)  # [2*rank, hidden_size]
        else:
            full_gate_up_proj0 = torch.empty((2*64, 128), dtype=dtype, device=device)
        dist.broadcast(full_gate_up_proj0, src=0, group=tp_pg)
        rank_per_shard = full_gate_up_proj0.shape[0] // tp_size  # 128 // 4 = 32
        start_idx = rank * rank_per_shard
        end_idx = (rank + 1) * rank_per_shard
        vanilla_layer.mlp.gate_up_proj0.weight.copy_(full_gate_up_proj0[start_idx:end_idx, :])
        
        # Gate/Up proj1 (RowLinear: shards input dim)
        for proj_name in ["gate", "up"]:
            if rank == 0:
                basic_proj = getattr(basic_layer.mlp, f"{proj_name}_proj")
                full_proj1 = basic_proj.cola_b.t()  # [intermediate_size, rank]
            else:
                full_proj1 = torch.empty((256, 64), dtype=dtype, device=device)
            dist.broadcast(full_proj1, src=0, group=tp_pg)
            rank_per_shard = full_proj1.shape[1] // tp_size  # 64 // 4 = 16
            start_idx = rank * rank_per_shard
            end_idx = (rank + 1) * rank_per_shard
            getattr(vanilla_layer.mlp, f"{proj_name}_proj1").weight.copy_(
                full_proj1[:, start_idx:end_idx]
            )
        
        # Down proj0 (ColumnLinear: shards output dim)
        if rank == 0:
            full_down_proj0 = basic_layer.mlp.down_proj.cola_a.t()  # [rank, intermediate_size]
        else:
            full_down_proj0 = torch.empty((64, 256), dtype=dtype, device=device)
        dist.broadcast(full_down_proj0, src=0, group=tp_pg)
        rank_per_shard = full_down_proj0.shape[0] // tp_size  # 64 // 4 = 16
        start_idx = rank * rank_per_shard
        end_idx = (rank + 1) * rank_per_shard
        vanilla_layer.mlp.down_proj0.weight.copy_(full_down_proj0[start_idx:end_idx, :])
        
        # Down proj1 (RowLinear: shards input dim)
        if rank == 0:
            full_down_proj1 = basic_layer.mlp.down_proj.cola_b.t()  # [hidden_size, rank]
        else:
            full_down_proj1 = torch.empty((128, 64), dtype=dtype, device=device)
        dist.broadcast(full_down_proj1, src=0, group=tp_pg)
        rank_per_shard = full_down_proj1.shape[1] // tp_size  # 64 // 4 = 16
        start_idx = rank * rank_per_shard
        end_idx = (rank + 1) * rank_per_shard
        vanilla_layer.mlp.down_proj1.weight.copy_(full_down_proj1[:, start_idx:end_idx])
        
        # Layer norms (not sharded, just broadcast)
        if rank == 0:
            vanilla_layer.input_layernorm.weight.copy_(basic_layer.input_layernorm.weight)
            vanilla_layer.post_attention_layernorm.weight.copy_(basic_layer.post_attention_layernorm.weight)
        dist.broadcast(vanilla_layer.input_layernorm.weight.data, src=0, group=tp_pg)
        dist.broadcast(vanilla_layer.post_attention_layernorm.weight.data, src=0, group=tp_pg)


def gather_vanilla_output(vanilla_out, tp_pg, hidden_size, rank):
    """
    Gather vanilla output from all TP ranks to reconstruct full tensor on rank 0.
    
    Note on TP Output Sharding:
    - In VanillaCoLA, the final decoder layer output should be full (not sharded)
      because the last projection (MLP down_proj1) is a RowLinear which performs all-reduce.
    - However, intermediate outputs might be sharded, so we check the shape to be safe.
    - If output is sharded (last dim < hidden_size), we gather from all TP ranks.
    - If output is already full (all-reduced), we just use rank 0's copy.
    
    Args:
        vanilla_out: Output tensor from vanilla layer [seq, batch, hidden_size_local or hidden_size]
        tp_pg: Tensor parallel process group
        hidden_size: Full hidden size
        rank: Current rank
    
    Returns:
        Full output tensor on rank 0, None on other ranks
    """
    seq_len, batch_size = vanilla_out.shape[:2]
    
    # Check if output is sharded (last dim < hidden_size)
    if vanilla_out.shape[-1] < hidden_size:
        # Output is sharded, need to gather
        # Create full tensor on rank 0
        if rank == 0:
            full_out = torch.empty(
                (seq_len, batch_size, hidden_size),
                dtype=vanilla_out.dtype,
                device=vanilla_out.device
            )
        else:
            full_out = None
        
        # Gather sharded outputs
        dist.all_gather_into_tensor(
            full_out,
            vanilla_out.contiguous(),
            group=tp_pg
        )
        
        return full_out
    else:
        # Output is already full (all-reduced), just return rank 0's copy
        if rank == 0:
            return vanilla_out
        else:
            return None


def test_layer_parity(layer_idx: int, tp_size: int = 4):
    """
    Test parity between Basic (TP=1) and Vanilla (TP=4) for a given layer index.
    
    Args:
        layer_idx: Layer index to test (0 or last layer)
        tp_size: Tensor parallel size (should be 4)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size != tp_size:
        if rank == 0:
            print(f"ERROR: World size {world_size} != TP size {tp_size}")
        return False
    
    # Set deterministic seeds (same on all ranks)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create configs
    basic_config, vanilla_config = create_test_configs()
    
    # Create TP4 parallel context for vanilla
    vanilla_parallel_context = ParallelContext(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    
    vanilla_parallel_config = ParallelismArgs(
        dp=1,
        tp=tp_size,
        pp=1,
        expert_parallel_size=1,
        recompute_layer=False,
    )
    
    # Create a dummy process group for Basic layer (TP=1)
    # Since Basic doesn't actually use TP (it's the reference implementation),
    # we can just use the main process group or create a minimal group
    # The simplest approach: use the main process group since Basic won't use TP features
    basic_tp_pg = dist.group.WORLD
    
    basic_parallel_config = ParallelismArgs(
        dp=1,
        tp=1,
        pp=1,
        expert_parallel_size=1,
        recompute_layer=False,
    )
    
    try:
        # Instantiate Basic layer (TP=1, only on rank 0)
        basic_layer = None
        if rank == 0:
            basic_layer = BasicColaLlamaDecoderLayer(
                config=basic_config,
                parallel_config=basic_parallel_config,
                tp_pg=basic_tp_pg,
                layer_idx=layer_idx,
            ).cuda().eval()
        
        # Instantiate Vanilla layer (TP=4, on all ranks)
        vanilla_layer = VanillaLlamaDecoderLayer(
            config=vanilla_config,
            parallel_config=vanilla_parallel_config,
            tp_pg=vanilla_parallel_context.tp_pg,
            layer_idx=layer_idx,
        ).cuda().eval()
        
        # Convert to bfloat16 before weight distribution
        # This ensures all weight copying happens in the target dtype
        if rank == 0:
            basic_layer = basic_layer.to(dtype=torch.bfloat16)
        vanilla_layer = vanilla_layer.to(dtype=torch.bfloat16)
        
        # Distribute Basic weights to Vanilla across all TP ranks
        # This function handles TP sharding correctly by extracting appropriate shards for each rank
        distribute_basic_weights_to_vanilla_tp(
            basic_layer,
            vanilla_layer,
            vanilla_parallel_context.tp_pg,
            rank,
            tp_size
        )
        
        # Create deterministic input on rank 0, then broadcast
        seq_len = 32
        batch_size = 2
        hidden_size = basic_config.hidden_size
        input_dtype = torch.bfloat16
        
        if rank == 0:
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            hidden_states = torch.randn(
                seq_len, batch_size, hidden_size,
                dtype=input_dtype,
                device="cuda"
            )
            sequence_mask = torch.ones(
                batch_size, seq_len,
                dtype=torch.bool,
                device="cuda"
            )
        else:
            hidden_states = torch.empty(
                seq_len, batch_size, hidden_size,
                dtype=input_dtype,
                device="cuda"
            )
            sequence_mask = torch.empty(
                batch_size, seq_len,
                dtype=torch.bool,
                device="cuda"
            )
        
        # Broadcast inputs from rank 0 to all ranks
        dist.broadcast(hidden_states, src=0, group=vanilla_parallel_context.tp_pg)
        dist.broadcast(sequence_mask, src=0, group=vanilla_parallel_context.tp_pg)
        
        # Run forward passes
        with torch.no_grad():
            # Vanilla forward (on all ranks)
            vanilla_output = vanilla_layer(
                hidden_states=hidden_states,
                sequence_mask=sequence_mask
            )
            vanilla_hidden = vanilla_output["hidden_states"]
            
            # Basic forward (only on rank 0)
            basic_hidden = None
            if rank == 0:
                basic_output = basic_layer(
                    hidden_states=hidden_states,
                    sequence_mask=sequence_mask
                )
                basic_hidden = basic_output["hidden_states"]
            
            # Gather vanilla output if needed (to reconstruct full tensor on rank 0)
            vanilla_hidden_full = gather_vanilla_output(
                vanilla_hidden,
                vanilla_parallel_context.tp_pg,
                hidden_size,
                rank
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
        
    finally:
        # Clean up parallel contexts
        vanilla_parallel_context.destroy()
        # Note: basic_tp_pg is a process group, not a ParallelContext, so we don't need to destroy it


def main():
    """Main test function"""
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
        print("=" * 80)
        print(f"World size: {world_size}, TP size: 4")
        print()
    
    # Test layer 0
    if rank == 0:
        print("Testing layer_idx=0 (first layer)...")
    success_0 = test_layer_parity(layer_idx=0, tp_size=4)
    
    # Test last layer
    if rank == 0:
        print("\nTesting layer_idx=1 (last layer, num_hidden_layers=2)...")
    success_last = test_layer_parity(layer_idx=1, tp_size=4)  # num_hidden_layers=2, so last is 1
    
    # Final summary
    if rank == 0:
        print("\n" + "=" * 80)
        if success_0 and success_last:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed!")
        print("=" * 80)
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

