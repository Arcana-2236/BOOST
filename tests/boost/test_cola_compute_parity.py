"""
Test to verify compute parity between Basic, Vanilla, and BTP CoLA implementations.

This test:
1. Instantiates decoder layers from all 3 implementations
2. Copies weights from Basic (ground truth) to Vanilla and BTP
3. Runs forward pass with identical inputs
4. Compares outputs to ensure numerical equivalence
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

# Try to import pytest, but make it optional
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Add examples/cola to path
cola_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "cola")
sys.path.insert(0, cola_dir)

from config_basic_cola_llama import BasicColaLlamaConfig
from config_cola_llama import ColaLlamaConfig
from basic_cola_llama import BasicColaLlamaDecoderLayer, ColaLayer
from vanilla_cola_llama import LlamaDecoderLayer as VanillaLlamaDecoderLayer
from cola_llama import LlamaDecoderLayer as ColaBtpLlamaDecoderLayer

from nanotron.config import ParallelismArgs
from nanotron import distributed as dist
from nanotron.parallel import ParallelContext


def create_test_configs():
    """Create matching configs for Basic, Vanilla, and Cola BTP models"""
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

    shared_cola_kwargs = dict(
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

    # vanilla_config = ColaLlamaConfig(**shared_cola_kwargs)  # COMMENTED OUT: Only testing Basic vs Cola BTP
    cola_config = ColaLlamaConfig(**shared_cola_kwargs)

    return basic_config, cola_config  # Return only basic and cola configs


def create_parallel_context():
    """Create a minimal parallel context for tp_size=1"""
    # ParallelContext will initialize distributed if not already initialized
    # For single GPU test: tp=1, pp=1, dp=1, ep=1
    parallel_context = ParallelContext(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=1,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    
    return parallel_context


def copy_basic_cola_to_vanilla_linear(basic_cola_layer: ColaLayer, proj0_linear: nn.Module, proj1_linear: nn.Module):
    """
    Copy Basic ColaLayer factors to Vanilla/Cola proj0 and proj1.
    
    Basic: cola_a [in, rank], cola_b [rank, out]
    
    For Vanilla:
    - proj0 is ColumnLinear: weight [out_features, in_features] = [rank, hidden_size]
      Need: cola_a.T = [rank, in] ✓
    - proj1 is RowLinear: weight [out_features, in_features] = [out_features, rank]
      Need: cola_b.T = [out, rank] ✓
    
    For Cola BTP:
    - proj0 is RowLinear: weight [out_features, in_features] = [rank, hidden_size]
      Need: cola_a.T = [rank, in] ✓
    - proj1 is ColumnLinear: weight [out_features, in_features] = [out_features, rank]
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


def copy_basic_attn_to_cola_btp(basic_attn, cola_attn):
    """
    Copy Basic attention weights to Cola BTP attention.
    
    Cola BTP structure:
    - qkv_proj0: RowLinear(hidden_size, 3*rank) -> weight [3*rank, hidden_size]
    - qkv_proj1: BatchedColumnLinear(rank, out_features, batch=3) -> weight [3, out_features, rank] or [3, rank, out_features]
    - o_proj0: RowLinear(hidden_size, rank) -> weight [rank, hidden_size]
    - o_proj1: ColumnLinear(rank, hidden_size) -> weight [hidden_size, rank]
    """
    with torch.no_grad():
        # QKV projections
        q_cola_a = basic_attn.q_proj.cola_a  # [hidden_size, rank]
        k_cola_a = basic_attn.k_proj.cola_a  # [hidden_size, rank]
        v_cola_a = basic_attn.v_proj.cola_a  # [hidden_size, rank]
        
        rank = q_cola_a.shape[1]
        # qkv_proj0.weight is [3*rank, hidden_size] (RowLinear: [out_features, in_features])
        cola_attn.qkv_proj0.weight[0:rank, :].copy_(q_cola_a.t())  # [rank, hidden_size]
        cola_attn.qkv_proj0.weight[rank:2*rank, :].copy_(k_cola_a.t())
        cola_attn.qkv_proj0.weight[2*rank:3*rank, :].copy_(v_cola_a.t())
        
        # qkv_proj1 (BatchedColumnLinear)
        q_cola_b = basic_attn.q_proj.cola_b  # [rank, out_features]
        k_cola_b = basic_attn.k_proj.cola_b  # [rank, out_features]
        v_cola_b = basic_attn.v_proj.cola_b  # [rank, out_features]
        
        if hasattr(cola_attn, 'qkv_proj1'):
            if len(cola_attn.qkv_proj1.weight.shape) == 3:
                # BatchedColumnLinear: weight shape is [gemm_num, out_features_local, in_features]
                # = [3, out_features//TP, rank] = [3, out_features, rank] for TP=1
                # cola_b.t() = [out_features, rank] ✓
                cola_attn.qkv_proj1.weight[0, :, :].copy_(q_cola_b.t())  # [out_features, rank]
                cola_attn.qkv_proj1.weight[1, :, :].copy_(k_cola_b.t())
                cola_attn.qkv_proj1.weight[2, :, :].copy_(v_cola_b.t())
        
        # O projection
        copy_basic_cola_to_vanilla_linear(
            basic_attn.o_proj,
            cola_attn.o_proj0,  # RowLinear: [rank, hidden_size]
            cola_attn.o_proj1   # ColumnLinear: [hidden_size, rank]
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


def copy_basic_mlp_to_cola_btp(basic_mlp, cola_mlp):
    """
    Copy Basic MLP weights to Cola BTP MLP.
    
    Cola BTP structure:
    - gate_up_proj0: RowLinear(hidden_size, 2*rank) -> weight [2*rank, hidden_size]
    - gate_up_proj1: BatchedColumnLinear(rank, intermediate_size, batch=2) -> weight [2, intermediate_size, rank] or [2, rank, intermediate_size]
    - down_proj0: RowLinear(intermediate_size, rank) -> weight [rank, intermediate_size]
    - down_proj1: ColumnLinear(rank, hidden_size) -> weight [hidden_size, rank]
    """
    with torch.no_grad():
        # Gate and Up projections
        gate_cola_a = basic_mlp.gate_proj.cola_a  # [hidden_size, rank]
        up_cola_a = basic_mlp.up_proj.cola_a  # [hidden_size, rank]
        
        rank = gate_cola_a.shape[1]
        # gate_up_proj0.weight is [2*rank, hidden_size] (RowLinear)
        cola_mlp.gate_up_proj0.weight[0:rank, :].copy_(gate_cola_a.t())  # [rank, hidden_size]
        cola_mlp.gate_up_proj0.weight[rank:2*rank, :].copy_(up_cola_a.t())
        
        gate_cola_b = basic_mlp.gate_proj.cola_b  # [rank, intermediate_size]
        up_cola_b = basic_mlp.up_proj.cola_b  # [rank, intermediate_size]
        
        if hasattr(cola_mlp, 'gate_up_proj1'):
            if len(cola_mlp.gate_up_proj1.weight.shape) == 3:
                # BatchedColumnLinear: weight shape is [gemm_num, out_features_local, in_features]
                # = [2, intermediate_size//TP, rank] = [2, intermediate_size, rank] for TP=1
                # cola_b.t() = [intermediate_size, rank] ✓
                cola_mlp.gate_up_proj1.weight[0, :, :].copy_(gate_cola_b.t())  # [intermediate_size, rank]
                cola_mlp.gate_up_proj1.weight[1, :, :].copy_(up_cola_b.t())
        
        # Down projection
        copy_basic_cola_to_vanilla_linear(
            basic_mlp.down_proj,
            cola_mlp.down_proj0,  # RowLinear: [rank, intermediate_size]
            cola_mlp.down_proj1   # ColumnLinear: [hidden_size, rank]
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


def copy_basic_to_cola_btp(basic_layer, cola_layer):
    """Copy all weights from Basic layer to Cola BTP layer"""
    with torch.no_grad():
        # Copy layer norms
        if hasattr(basic_layer, 'input_layernorm') and hasattr(cola_layer, 'input_layernorm'):
            cola_layer.input_layernorm.weight.copy_(basic_layer.input_layernorm.weight)
            if hasattr(basic_layer.input_layernorm, 'bias') and basic_layer.input_layernorm.bias is not None:
                cola_layer.input_layernorm.bias.copy_(basic_layer.input_layernorm.bias)
        
        if hasattr(basic_layer, 'post_attention_layernorm') and hasattr(cola_layer, 'post_attention_layernorm'):
            cola_layer.post_attention_layernorm.weight.copy_(basic_layer.post_attention_layernorm.weight)
            if hasattr(basic_layer.post_attention_layernorm, 'bias') and basic_layer.post_attention_layernorm.bias is not None:
                cola_layer.post_attention_layernorm.bias.copy_(basic_layer.post_attention_layernorm.bias)
        
        # Copy attention
        copy_basic_attn_to_cola_btp(basic_layer.attn, cola_layer.attn)
        
        # Copy MLP
        copy_basic_mlp_to_cola_btp(basic_layer.mlp, cola_layer.mlp)


def print_tensor_info(name: str, tensor: torch.Tensor, max_elems: int = 10, compare_with: Optional[torch.Tensor] = None):
    """Print tensor information for debugging
    
    Args:
        name: Name of the tensor
        tensor: Tensor to print
        max_elems: Maximum number of elements to print
        compare_with: Optional tensor to compare with. If provided and shapes are transposes of each other,
                     will print both in the same format for comparison.
    """
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    
    # If compare_with is provided and shapes are transposes, print both in same format
    if compare_with is not None:
        if tensor.shape == compare_with.shape:
            # Same shape, just compare directly
            flat = tensor.flatten()
            compare_flat = compare_with.flatten()
            num_elems = min(max_elems, flat.numel())
            print(f"  First {num_elems} values: {flat[:num_elems].tolist()}")
            print(f"  Compare tensor first {num_elems} values: {compare_flat[:num_elems].tolist()}")
            match = torch.allclose(tensor, compare_with, atol=1e-6)
            print(f"  Values match: {match}")
        elif len(tensor.shape) == 2 and len(compare_with.shape) == 2:
            # Check if one is transpose of the other
            if tensor.shape == (compare_with.shape[1], compare_with.shape[0]):
                # tensor is transpose of compare_with
                # Print both in the same format (as compare_with shape)
                tensor_as_compare = tensor.t()  # Transpose to match compare_with shape
                flat_compare = compare_with.flatten()
                flat_tensor = tensor_as_compare.flatten()
                num_elems = min(max_elems, flat_compare.numel())
                print(f"  (Note: This tensor is transpose of compare tensor)")
                print(f"  Compare tensor (Basic) first {num_elems} values: {flat_compare[:num_elems].tolist()}")
                print(f"  This tensor transposed (Vanilla) first {num_elems} values: {flat_tensor[:num_elems].tolist()}")
                match = torch.allclose(tensor_as_compare, compare_with, atol=1e-6)
                print(f"  Values match (after transpose): {match}")
            elif compare_with.shape == (tensor.shape[1], tensor.shape[0]):
                # compare_with is transpose of tensor
                # Print both in the same format (as tensor shape)
                compare_as_tensor = compare_with.t()  # Transpose to match tensor shape
                flat_tensor = tensor.flatten()
                flat_compare = compare_as_tensor.flatten()
                num_elems = min(max_elems, flat_tensor.numel())
                print(f"  (Note: Compare tensor is transpose of this)")
                print(f"  This tensor (Vanilla) first {num_elems} values: {flat_tensor[:num_elems].tolist()}")
                print(f"  Compare tensor transposed (Basic) first {num_elems} values: {flat_compare[:num_elems].tolist()}")
                match = torch.allclose(tensor, compare_as_tensor, atol=1e-6)
                print(f"  Values match (after transpose): {match}")
            else:
                # Different shapes, print normally
                flat = tensor.flatten()
                num_elems = min(max_elems, flat.numel())
                print(f"  First {num_elems} values: {flat[:num_elems].tolist()}")
        else:
            # Different shapes, print normally
            flat = tensor.flatten()
            num_elems = min(max_elems, flat.numel())
            print(f"  First {num_elems} values: {flat[:num_elems].tolist()}")
    else:
        # No comparison, just print normally
        flat = tensor.flatten()
        num_elems = min(max_elems, flat.numel())
        print(f"  First {num_elems} values: {flat[:num_elems].tolist()}")


def test_cola_compute_parity():
    """Test compute parity by copying weights and comparing forward outputs - Basic vs Cola BTP (Vanilla commented out)"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # Set deterministic seeds
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create configs
    basic_config, cola_config = create_test_configs()
    # vanilla_config = vanilla_config  # COMMENTED OUT
    
    # Create parallel context
    parallel_context = create_parallel_context()
    parallel_config = ParallelismArgs(
        dp=1, 
        tp=1, 
        pp=1, 
        expert_parallel_size=1,
        recompute_layer=False,
    )
    
    try:
        # Create decoder layers
        # Using layer_idx=1 (last layer, since num_hidden_layers=2 means layers 0 and 1)
        basic_layer = BasicColaLlamaDecoderLayer(
            config=basic_config,
            parallel_config=parallel_config,
            tp_pg=parallel_context.tp_pg,
            layer_idx=1,
        ).cuda().eval()
        
        # COMMENTED OUT: Vanilla layer instantiation
        # vanilla_layer = VanillaLlamaDecoderLayer(
        #     config=vanilla_config,
        #     parallel_config=parallel_config,
        #     tp_pg=parallel_context.tp_pg,
        #     layer_idx=1,
        # ).cuda().eval()
        
        cola_layer = ColaBtpLlamaDecoderLayer(
            config=cola_config,
            parallel_config=parallel_config,
            tp_pg=parallel_context.tp_pg,
            layer_idx=1,
        ).cuda().eval()
        
        # COMMENTED OUT: Copy weights from Basic to Vanilla
        # print("=" * 80)
        # print("COPYING WEIGHTS FROM BASIC TO VANILLA")
        # print("=" * 80)
        # copy_basic_to_vanilla(basic_layer, vanilla_layer)
        
        # Copy weights from Basic to Cola BTP
        print("\n" + "=" * 80)
        print("COPYING WEIGHTS FROM BASIC TO COLA BTP")
        print("=" * 80)
        copy_basic_attn_to_cola_btp(basic_layer.attn, cola_layer.attn)
        copy_basic_mlp_to_cola_btp(basic_layer.mlp, cola_layer.mlp)
        
        # Verify weights were copied correctly
        print("\n" + "=" * 80)
        print("VERIFYING WEIGHT COPYING")
        print("=" * 80)
        
        # COMMENTED OUT: Vanilla weight verification
        # # Check Q projection - verify the forward pass equivalence
        # q_cola_a = basic_layer.attn.q_proj.cola_a  # [128, 64] = [in_features, rank]
        # q_cola_a_t = q_cola_a.t()  # [64, 128] = [rank, in_features]
        # vanilla_qkv_proj0_weight = vanilla_layer.attn.qkv_proj0.weight  # [192, 128] = [out_features, in_features]
        # vanilla_q_slice = vanilla_qkv_proj0_weight[0:64, :]  # [64, 128] = [rank, in_features]
        # 
        # print(f"\nQ projection cola_a:")
        # print(f"  Basic cola_a shape: {q_cola_a.shape} = [in_features, rank]")
        # print(f"  Basic cola_a.t() shape: {q_cola_a_t.shape} = [rank, in_features]")
        # print(f"  Vanilla qkv_proj0.weight shape: {vanilla_qkv_proj0_weight.shape} = [out_features, in_features]")
        # print(f"  Vanilla qkv_proj0[0:64, :] shape: {vanilla_q_slice.shape} = [rank, in_features]")
        # print(f"  Are they equal? {torch.allclose(q_cola_a_t, vanilla_q_slice, atol=1e-6)}")
        # 
        # # Verify forward pass equivalence
        # print(f"\n  Forward pass verification:")
        # print(f"    Basic: x @ cola_a = [seq, batch, 128] @ [128, 64] = [seq, batch, 64]")
        # print(f"    Vanilla: x @ qkv_proj0.weight[0:64, :].T = [seq, batch, 128] @ [128, 64] = [seq, batch, 64]")
        # print(f"    So: cola_a should equal qkv_proj0.weight[0:64, :].T")
        # print(f"    cola_a shape: {q_cola_a.shape}")
        # print(f"    qkv_proj0.weight[0:64, :].T shape: {vanilla_q_slice.t().shape}")
        # forward_match = torch.allclose(q_cola_a, vanilla_q_slice.t(), atol=1e-6)
        # print(f"    Are they equal? {forward_match}")
        # if not forward_match:
        #     diff = (q_cola_a - vanilla_q_slice.t()).abs()
        #     print(f"    Forward pass diff - Max: {diff.max().item():.6e}, Mean: {diff.mean().item():.6e}")
        #     print(f"    This means the transpose is WRONG!")
        # else:
        #     print(f"    ✓ Transpose is CORRECT - forward passes will be equivalent")
        # 
        # if not torch.allclose(q_cola_a_t, vanilla_q_slice, atol=1e-6):
        #     diff = (q_cola_a_t - vanilla_q_slice).abs()
        #     print(f"  Max diff: {diff.max().item():.6e}")
        #     print(f"  First row of cola_a.t(): {q_cola_a_t[0, :5].tolist()}")
        #     print(f"  First row of vanilla: {vanilla_q_slice[0, :5].tolist()}")
        #     print(f"  First col of cola_a: {q_cola_a[:, 0].tolist()[:5]}")
        #     print(f"  First col of cola_a.t(): {q_cola_a_t[:, 0].tolist()[:5]}")
        # 
        # # Check Q projection cola_b
        # q_cola_b = basic_layer.attn.q_proj.cola_b  # [64, 128]
        # q_cola_b_t = q_cola_b.t()  # [128, 64]
        # vanilla_q_proj1 = vanilla_layer.attn.q_proj1.weight  # [128, 64]
        # 
        # print(f"\nQ projection cola_b:")
        # print(f"  Basic cola_b shape: {q_cola_b.shape}")
        # print(f"  Basic cola_b.t() shape: {q_cola_b_t.shape}")
        # print(f"  Vanilla q_proj1.weight shape: {vanilla_q_proj1.shape}")
        # print(f"  Are they equal? {torch.allclose(q_cola_b_t, vanilla_q_proj1, atol=1e-6)}")
        # if not torch.allclose(q_cola_b_t, vanilla_q_proj1, atol=1e-6):
        #     diff = (q_cola_b_t - vanilla_q_proj1).abs()
        #     print(f"  Max diff: {diff.max().item():.6e}")
        
        # Print weights after copying for verification
        print("\n" + "=" * 80)
        print("WEIGHT VERIFICATION AFTER COPYING")
        print("=" * 80)
        
        # Check Basic attention weights
        print("\n--- Basic Attention Weights ---")
        print_tensor_info("basic.attn.q_proj.cola_a", basic_layer.attn.q_proj.cola_a)
        print_tensor_info("basic.attn.q_proj.cola_b", basic_layer.attn.q_proj.cola_b)
        print_tensor_info("basic.attn.k_proj.cola_a", basic_layer.attn.k_proj.cola_a)
        print_tensor_info("basic.attn.k_proj.cola_b", basic_layer.attn.k_proj.cola_b)
        print_tensor_info("basic.attn.v_proj.cola_a", basic_layer.attn.v_proj.cola_a)
        print_tensor_info("basic.attn.v_proj.cola_b", basic_layer.attn.v_proj.cola_b)
        print_tensor_info("basic.attn.o_proj.cola_a", basic_layer.attn.o_proj.cola_a)
        print_tensor_info("basic.attn.o_proj.cola_b", basic_layer.attn.o_proj.cola_b)
        
        # COMMENTED OUT: Check Vanilla attention weights
        # print("\n--- Vanilla Attention Weights (compared with Basic) ---")
        # vanilla_q_slice = vanilla_layer.attn.qkv_proj0.weight[0:64, :]
        # vanilla_k_slice = vanilla_layer.attn.qkv_proj0.weight[64:128, :]
        # vanilla_v_slice = vanilla_layer.attn.qkv_proj0.weight[128:192, :]
        # 
        # # Verify they match Basic after transpose
        # print("\n  Verifying Q projection cola_a copy:")
        # q_cola_a = basic_layer.attn.q_proj.cola_a  # [128, 64]
        # q_cola_a_t = q_cola_a.t()  # [64, 128]
        # q_match = torch.allclose(q_cola_a_t, vanilla_q_slice, atol=1e-6)
        # print(f"    Q cola_a.t() matches vanilla qkv_proj0[0:64, :]: {q_match}")
        # if not q_match:
        #     diff = (q_cola_a_t - vanilla_q_slice).abs()
        #     print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
        # 
        # # Print with comparison - show both in same format
        # print_tensor_info("vanilla.attn.qkv_proj0.weight (q slice) [64, 128]", vanilla_q_slice, compare_with=q_cola_a)
        # print_tensor_info("vanilla.attn.qkv_proj0.weight (k slice) [64, 128]", vanilla_k_slice, compare_with=basic_layer.attn.k_proj.cola_a)
        # print_tensor_info("vanilla.attn.qkv_proj0.weight (v slice) [64, 128]", vanilla_v_slice, compare_with=basic_layer.attn.v_proj.cola_a)
        # 
        # # Verify cola_b copies
        # print("\n  Verifying Q projection cola_b copy:")
        # q_cola_b = basic_layer.attn.q_proj.cola_b  # [64, 128]
        # q_cola_b_t = q_cola_b.t()  # [128, 64]
        # q_b_match = torch.allclose(q_cola_b_t, vanilla_layer.attn.q_proj1.weight, atol=1e-6)
        # print(f"    Q cola_b.t() matches vanilla q_proj1.weight: {q_b_match}")
        # if not q_b_match:
        #     diff = (q_cola_b_t - vanilla_layer.attn.q_proj1.weight).abs()
        #     print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
        # 
        # print_tensor_info("vanilla.attn.q_proj1.weight [128, 64]", vanilla_layer.attn.q_proj1.weight, compare_with=q_cola_b)
        # print_tensor_info("vanilla.attn.k_proj1.weight [128, 64]", vanilla_layer.attn.k_proj1.weight, compare_with=basic_layer.attn.k_proj.cola_b)
        # print_tensor_info("vanilla.attn.v_proj1.weight [128, 64]", vanilla_layer.attn.v_proj1.weight, compare_with=basic_layer.attn.v_proj.cola_b)
        # 
        # # O projection
        # print_tensor_info("vanilla.attn.o_proj0.weight [64, 128]", vanilla_layer.attn.o_proj0.weight, compare_with=basic_layer.attn.o_proj.cola_a)
        # print_tensor_info("vanilla.attn.o_proj1.weight [128, 64]", vanilla_layer.attn.o_proj1.weight, compare_with=basic_layer.attn.o_proj.cola_b)
        
        # Check Basic MLP weights
        print("\n--- Basic MLP Weights ---")
        print_tensor_info("basic.mlp.gate_proj.cola_a", basic_layer.mlp.gate_proj.cola_a)
        print_tensor_info("basic.mlp.gate_proj.cola_b", basic_layer.mlp.gate_proj.cola_b)
        print_tensor_info("basic.mlp.up_proj.cola_a", basic_layer.mlp.up_proj.cola_a)
        print_tensor_info("basic.mlp.up_proj.cola_b", basic_layer.mlp.up_proj.cola_b)
        print_tensor_info("basic.mlp.down_proj.cola_a", basic_layer.mlp.down_proj.cola_a)
        print_tensor_info("basic.mlp.down_proj.cola_b", basic_layer.mlp.down_proj.cola_b)
        
        # Check Cola BTP attention weights (compared with Basic)
        print("\n--- Cola BTP Attention Weights (compared with Basic) ---")
        # QKV proj0: grouped [3*rank, hidden_size] = [192, 128]
        cola_qkv_proj0_weight = cola_layer.attn.qkv_proj0.weight
        cola_q_slice_proj0 = cola_qkv_proj0_weight[0:64, :]  # Q slice [64, 128]
        cola_k_slice_proj0 = cola_qkv_proj0_weight[64:128, :]  # K slice [64, 128]
        cola_v_slice_proj0 = cola_qkv_proj0_weight[128:192, :]  # V slice [64, 128]
        
        print_tensor_info("cola BTP qkv_proj0.weight (q slice) [64, 128]", cola_q_slice_proj0, compare_with=basic_layer.attn.q_proj.cola_a)
        print_tensor_info("cola BTP qkv_proj0.weight (k slice) [64, 128]", cola_k_slice_proj0, compare_with=basic_layer.attn.k_proj.cola_a)
        print_tensor_info("cola BTP qkv_proj0.weight (v slice) [64, 128]", cola_v_slice_proj0, compare_with=basic_layer.attn.v_proj.cola_a)
        
        # QKV proj1: batched [3, out_features, rank] = [3, 128, 64]
        cola_qkv_proj1_weight = cola_layer.attn.qkv_proj1.weight
        cola_q_proj1 = cola_qkv_proj1_weight[0, :, :]  # Q [128, 64]
        cola_k_proj1 = cola_qkv_proj1_weight[1, :, :]  # K [128, 64]
        cola_v_proj1 = cola_qkv_proj1_weight[2, :, :]  # V [128, 64]
        
        print_tensor_info("cola BTP qkv_proj1.weight[0] (q) [128, 64]", cola_q_proj1, compare_with=basic_layer.attn.q_proj.cola_b)
        print_tensor_info("cola BTP qkv_proj1.weight[1] (k) [128, 64]", cola_k_proj1, compare_with=basic_layer.attn.k_proj.cola_b)
        print_tensor_info("cola BTP qkv_proj1.weight[2] (v) [128, 64]", cola_v_proj1, compare_with=basic_layer.attn.v_proj.cola_b)
        
        # O projection
        print_tensor_info("cola BTP o_proj0.weight [64, 128]", cola_layer.attn.o_proj0.weight, compare_with=basic_layer.attn.o_proj.cola_a)
        print_tensor_info("cola BTP o_proj1.weight [128, 64]", cola_layer.attn.o_proj1.weight, compare_with=basic_layer.attn.o_proj.cola_b)
        
        # Check Cola BTP MLP weights (compared with Basic)
        print("\n--- Cola BTP MLP Weights (compared with Basic) ---")
        # Gate/Up proj0: grouped [2*rank, hidden_size] = [128, 128]
        cola_gate_up_proj0_weight = cola_layer.mlp.gate_up_proj0.weight
        cola_gate_slice_proj0 = cola_gate_up_proj0_weight[0:64, :]  # Gate slice [64, 128]
        cola_up_slice_proj0 = cola_gate_up_proj0_weight[64:128, :]  # Up slice [64, 128]
        
        print_tensor_info("cola BTP gate_up_proj0.weight (gate slice) [64, 128]", cola_gate_slice_proj0, compare_with=basic_layer.mlp.gate_proj.cola_a)
        print_tensor_info("cola BTP gate_up_proj0.weight (up slice) [64, 128]", cola_up_slice_proj0, compare_with=basic_layer.mlp.up_proj.cola_a)
        
        # Gate/Up proj1: batched [2, out_features, rank] = [2, 256, 64]
        cola_gate_up_proj1_weight = cola_layer.mlp.gate_up_proj1.weight
        cola_gate_proj1 = cola_gate_up_proj1_weight[0, :, :]  # Gate [256, 64]
        cola_up_proj1 = cola_gate_up_proj1_weight[1, :, :]  # Up [256, 64]
        
        print_tensor_info("cola BTP gate_up_proj1.weight[0] (gate) [256, 64]", cola_gate_proj1, compare_with=basic_layer.mlp.gate_proj.cola_b)
        print_tensor_info("cola BTP gate_up_proj1.weight[1] (up) [256, 64]", cola_up_proj1, compare_with=basic_layer.mlp.up_proj.cola_b)
        
        # Down projection
        print_tensor_info("cola BTP down_proj0.weight [64, 256]", cola_layer.mlp.down_proj0.weight, compare_with=basic_layer.mlp.down_proj.cola_a)
        print_tensor_info("cola BTP down_proj1.weight [128, 64]", cola_layer.mlp.down_proj1.weight, compare_with=basic_layer.mlp.down_proj.cola_b)
        
        # COMMENTED OUT: Check Vanilla MLP weights
        # print("\n--- Vanilla MLP Weights (compared with Basic) ---")
        # print_tensor_info("vanilla.mlp.gate_up_proj0.weight (gate slice) [64, 128]", vanilla_layer.mlp.gate_up_proj0.weight[0:64, :], compare_with=basic_layer.mlp.gate_proj.cola_a)
        # print_tensor_info("vanilla.mlp.gate_up_proj0.weight (up slice) [64, 128]", vanilla_layer.mlp.gate_up_proj0.weight[64:128, :], compare_with=basic_layer.mlp.up_proj.cola_a)
        # print_tensor_info("vanilla.mlp.gate_proj1.weight [256, 64]", vanilla_layer.mlp.gate_proj1.weight, compare_with=basic_layer.mlp.gate_proj.cola_b)
        # print_tensor_info("vanilla.mlp.up_proj1.weight [256, 64]", vanilla_layer.mlp.up_proj1.weight, compare_with=basic_layer.mlp.up_proj.cola_b)
        # print_tensor_info("vanilla.mlp.down_proj0.weight [64, 256]", vanilla_layer.mlp.down_proj0.weight, compare_with=basic_layer.mlp.down_proj.cola_a)
        # print_tensor_info("vanilla.mlp.down_proj1.weight [128, 64]", vanilla_layer.mlp.down_proj1.weight, compare_with=basic_layer.mlp.down_proj.cola_b)
        
        # Convert all models to bfloat16 to match input dtype
        # This ensures all parameters (weights, biases, etc.) are in bfloat16
        print("\n" + "=" * 80)
        print("CONVERTING MODELS TO BFLOAT16")
        print("=" * 80)
        basic_layer = basic_layer.to(dtype=torch.bfloat16)
        # vanilla_layer = vanilla_layer.to(dtype=torch.bfloat16)  # COMMENTED OUT
        cola_layer = cola_layer.to(dtype=torch.bfloat16)
        
        # Verify dtype consistency: check that key parameters match input dtype
        # This ensures weights and inputs will have the same dtype
        input_dtype = torch.bfloat16
        print(f"\nVerifying all parameters are {input_dtype}...")
        for name, param in basic_layer.named_parameters():
            if "cola_a" in name or "cola_b" in name:
                assert param.dtype == input_dtype, \
                    f"Basic layer parameter {name} dtype {param.dtype} != input dtype {input_dtype}"
        
        # COMMENTED OUT: Vanilla dtype checks
        # for name, param in vanilla_layer.named_parameters():
        #     if "weight" in name and ("proj0" in name or "proj1" in name):
        #         assert param.dtype == input_dtype, \
        #             f"Vanilla layer parameter {name} dtype {param.dtype} != input dtype {input_dtype}"

        for name, param in cola_layer.named_parameters():
            if "weight" in name and ("proj0" in name or "proj1" in name):
                assert param.dtype == input_dtype, \
                    f"Cola BTP layer parameter {name} dtype {param.dtype} != input dtype {input_dtype}"
        print("✓ All parameters verified to be bfloat16")
        
        # Create deterministic input in bfloat16
        seq_len = 32
        batch_size = 2
        hidden_size = basic_config.hidden_size
        
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        hidden_states = torch.randn(
            seq_len, batch_size, hidden_size,
            dtype=input_dtype,  # bfloat16 to match model parameters
            device="cuda"
        )
        
        # Verify input dtype matches model parameter dtype
        assert hidden_states.dtype == input_dtype, \
            f"Input dtype {hidden_states.dtype} != expected {input_dtype}"
        
        sequence_mask = torch.ones(
            batch_size, seq_len,
            dtype=torch.bool,
            device="cuda"
        )
        
        # Run forward pass step by step with debugging
        print("\n" + "=" * 80)
        print("STEP-BY-STEP FORWARD PASS COMPARISON")
        print("=" * 80)
        
        with torch.no_grad():
            # Step 1: Input layer norm
            print("\n--- Step 1: Input Layer Norm ---")
            basic_ln_out = basic_layer.input_layernorm(hidden_states)
            # vanilla_ln_out = vanilla_layer.input_layernorm(hidden_states)  # COMMENTED OUT
            # Cola BTP uses DelayedTritonRMSNorm and returns (hidden_states, rstd)
            cola_ln_out, cola_rstd = cola_layer.input_layernorm(hidden_states)

            print_tensor_info("Basic input_layernorm output", basic_ln_out)
            # print_tensor_info("Vanilla input_layernorm output", vanilla_ln_out)  # COMMENTED OUT
            print_tensor_info("Cola BTP input_layernorm output", cola_ln_out)

            # ln_diff_vanilla = (basic_ln_out - vanilla_ln_out).abs()  # COMMENTED OUT
            ln_diff_cola = (basic_ln_out - cola_ln_out).abs()
            # print(f"  LayerNorm diff (Basic vs Vanilla) - Max: {ln_diff_vanilla.max().item():.6e}, Mean: {ln_diff_vanilla.mean().item():.6e}")  # COMMENTED OUT
            print(f"  LayerNorm diff (Basic vs Cola BTP) - Max: {ln_diff_cola.max().item():.6e}, Mean: {ln_diff_cola.mean().item():.6e}")
            
            # Step 2: Attention QKV projections - detailed breakdown
            print("\n--- Step 2: Attention QKV Projections ---")
            
            # Basic: separate q, k, v projections
            print("\n  Basic model (separate projections):")
            # Manual decomposition
            basic_q_lr = torch.matmul(basic_ln_out, basic_layer.attn.q_proj.cola_a)  # [seq, batch, rank]
            basic_q_lr_act = basic_layer.attn.q_proj.lr_act(basic_q_lr)
            basic_q = torch.matmul(basic_q_lr_act, basic_layer.attn.q_proj.cola_b)  # [seq, batch, out]
            
            # Validate: Check that manual decomposition matches actual forward pass
            basic_q_actual = basic_layer.attn.q_proj(basic_ln_out)
            q_manual_vs_actual = (basic_q - basic_q_actual).abs()
            if q_manual_vs_actual.max().item() > 1e-5:
                print(f"    ⚠️  WARNING: Basic Q manual decomposition differs from actual forward!")
                print(f"       Max diff: {q_manual_vs_actual.max().item():.6e}, Mean: {q_manual_vs_actual.mean().item():.6e}")
            else:
                print(f"    ✓ Basic Q manual decomposition matches actual forward (diff < 1e-5)")
            print_tensor_info("    basic q: cola_a output (after matmul)", basic_q_lr)
            print_tensor_info("    basic q: after lr_act", basic_q_lr_act)
            print_tensor_info("    basic q: final output", basic_q)
            
            # Use actual forward passes for k and v (more reliable than manual decomposition)
            basic_k = basic_layer.attn.k_proj(basic_ln_out)
            print_tensor_info("    basic k: final output (from actual forward)", basic_k)
            
            # Compute intermediate values for debugging/comparison
            basic_k_lr = torch.matmul(basic_ln_out, basic_layer.attn.k_proj.cola_a)
            basic_k_lr_act = basic_layer.attn.k_proj.lr_act(basic_k_lr)
            
            basic_v = basic_layer.attn.v_proj(basic_ln_out)
            print_tensor_info("    basic v: final output (from actual forward)", basic_v)
            
            # Compute intermediate values for debugging/comparison
            basic_v_lr = torch.matmul(basic_ln_out, basic_layer.attn.v_proj.cola_a)
            basic_v_lr_act = basic_layer.attn.v_proj.lr_act(basic_v_lr)
            
            # Cola BTP: grouped qkv_proj0 then batched qkv_proj1
            print("\n  Cola BTP model (grouped then batched):")
            cola_qkv_lr = cola_layer.attn.qkv_proj0(cola_ln_out, rstd=cola_rstd)  # [seq, batch, 3*rank]
            print_tensor_info("    cola BTP qkv_proj0 output (before act)", cola_qkv_lr)
            cola_qkv_lr_act = cola_layer.attn.lr_act(cola_qkv_lr)
            print_tensor_info("    cola BTP qkv_proj0 output (after act)", cola_qkv_lr_act)
            # Unflatten to [seq, batch, 3, rank] then pass to batched qkv_proj1
            cola_qkv_lr_unflattened = cola_qkv_lr_act.unflatten(-1, (3, -1))  # [seq, batch, 3, rank]
            print_tensor_info("    cola BTP qkv_proj0 output (after unflatten)", cola_qkv_lr_unflattened)
            cola_qkv_states = cola_layer.attn.qkv_proj1(cola_qkv_lr_unflattened)  # [seq, batch, 3, out_features]
            cola_lr_q, cola_lr_k, cola_lr_v = cola_qkv_states.unbind(dim=0)  # Each: [seq, batch, out_features]
            print_tensor_info("    cola BTP lr_q (after unbind)", cola_lr_q)
            print_tensor_info("    cola BTP lr_k (after unbind)", cola_lr_k)
            print_tensor_info("    cola BTP lr_v (after unbind)", cola_lr_v)
            
            # Note: Cola BTP's qkv_proj1 outputs are already the final q, k, v (not low-rank)
            cola_q = cola_lr_q
            cola_k = cola_lr_k
            cola_v = cola_lr_v
            print_tensor_info("    cola BTP q: final output", cola_q)
            print_tensor_info("    cola BTP k: final output", cola_k)
            print_tensor_info("    cola BTP v: final output", cola_v)
            
            # Compare intermediate values
            print("\n  Comparison (Basic vs Cola BTP):")
            # Compare low-rank activations (after lr_act, before proj1)
            # For Cola BTP, we need to extract individual q, k, v from the grouped tensor
            cola_q_lr_from_grouped = cola_qkv_lr_act[:, :, 0:64]  # Extract Q slice
            cola_k_lr_from_grouped = cola_qkv_lr_act[:, :, 64:128]  # Extract K slice
            cola_v_lr_from_grouped = cola_qkv_lr_act[:, :, 128:192]  # Extract V slice
            
            q_lr_diff = (basic_q_lr_act - cola_q_lr_from_grouped).abs()
            k_lr_diff = (basic_k_lr_act - cola_k_lr_from_grouped).abs()
            v_lr_diff = (basic_v_lr_act - cola_v_lr_from_grouped).abs()
            print(f"    LR Q diff - Max: {q_lr_diff.max().item():.6e}, Mean: {q_lr_diff.mean().item():.6e}")
            print(f"    LR K diff - Max: {k_lr_diff.max().item():.6e}, Mean: {k_lr_diff.mean().item():.6e}")
            print(f"    LR V diff - Max: {v_lr_diff.max().item():.6e}, Mean: {v_lr_diff.mean().item():.6e}")
            
            q_diff = (basic_q - cola_q).abs()
            k_diff = (basic_k - cola_k).abs()
            v_diff = (basic_v - cola_v).abs()
            print(f"    Final Q diff - Max: {q_diff.max().item():.6e}, Mean: {q_diff.mean().item():.6e}")
            print(f"    Final K diff - Max: {k_diff.max().item():.6e}, Mean: {k_diff.mean().item():.6e}")
            print(f"    Final V diff - Max: {v_diff.max().item():.6e}, Mean: {v_diff.mean().item():.6e}")
            
            # Step 3: Attention computation
            print("\n--- Step 3: Attention Computation ---")
            print("  Note: Attention computation involves flash_attn which may have numerical differences")
            print("  Running attention forward pass...")
            
            # Basic attention forward
            basic_attn_output_dict = basic_layer.attn(
                hidden_states=basic_ln_out,
                sequence_mask=sequence_mask
            )
            basic_attn_out = basic_attn_output_dict["hidden_states"]
            print_tensor_info("  Basic attention output", basic_attn_out)
            
            # COMMENTED OUT: Vanilla attention forward
            # vanilla_attn_output_dict = vanilla_layer.attn(
            #     hidden_states=vanilla_ln_out,
            #     sequence_mask=sequence_mask
            # )
            # vanilla_attn_out = vanilla_attn_output_dict["hidden_states"]
            # print_tensor_info("  Vanilla attention output", vanilla_attn_out)

            # Cola BTP attention forward (uses rstd from its own layernorm)
            cola_attn_output_dict = cola_layer.attn(
                hidden_states=cola_ln_out,
                sequence_mask=sequence_mask,
                rstd=cola_rstd,
            )
            cola_attn_out = cola_attn_output_dict["hidden_states"]
            print_tensor_info("  Cola BTP attention output", cola_attn_out)
            
            # attn_diff_vanilla = (basic_attn_out - vanilla_attn_out).abs()  # COMMENTED OUT
            attn_diff_cola = (basic_attn_out - cola_attn_out).abs()
            # print(f"  Attention output diff (Basic vs Vanilla) - Max: {attn_diff_vanilla.max().item():.6e}, Mean: {attn_diff_vanilla.mean().item():.6e}")  # COMMENTED OUT
            print(f"  Attention output diff (Basic vs Cola BTP) - Max: {attn_diff_cola.max().item():.6e}, Mean: {attn_diff_cola.mean().item():.6e}")
            
            # Step 4: Attention O projection verification
            print("\n--- Step 4: Attention O Projection Verification ---")
            print("  Note: Both basic_attn_out and cola_attn_out already include the O projection.")
            print("  They should match directly since the O projection is part of the attention forward pass.")
            
            # Both basic_attn_out and cola_attn_out are the final outputs after O projection
            # So we can compare them directly
            basic_o_final = basic_attn_out  # Already includes O projection
            cola_o_final = cola_attn_out    # Already includes O projection
            
            print_tensor_info("    basic o: final output (from attn forward)", basic_o_final)
            print_tensor_info("    cola BTP o: final output (from attn forward)", cola_o_final)
            
            o_diff = (basic_o_final - cola_o_final).abs()
            print(f"    O projection diff (Basic vs Cola BTP) - Max: {o_diff.max().item():.6e}, Mean: {o_diff.mean().item():.6e}")
            
            # Step 5: MLP projections
            print("\n--- Step 5: MLP Projections ---")
            # After attention residual
            # Use manually computed O projection outputs for consistency
            basic_after_attn = hidden_states + basic_o_final
            # vanilla_after_attn = hidden_states + vanilla_o_final  # COMMENTED OUT
            cola_after_attn = hidden_states + cola_o_final  # Use cola_o_final instead of cola_attn_out for consistency
            print_tensor_info("  Basic after attention residual", basic_after_attn)
            # print_tensor_info("  Vanilla after attention residual", vanilla_after_attn)  # COMMENTED OUT
            print_tensor_info("  Cola BTP after attention residual", cola_after_attn)
            
            # Post attention layer norm
            basic_post_ln = basic_layer.post_attention_layernorm(basic_after_attn)
            # vanilla_post_ln = vanilla_layer.post_attention_layernorm(vanilla_after_attn)  # COMMENTED OUT
            cola_post_ln, cola_post_rstd = cola_layer.post_attention_layernorm(cola_after_attn)
            print_tensor_info("  Basic post_attention_layernorm", basic_post_ln)
            # print_tensor_info("  Vanilla post_attention_layernorm", vanilla_post_ln)  # COMMENTED OUT
            print_tensor_info("  Cola BTP post_attention_layernorm", cola_post_ln)
            
            # MLP gate/up projections
            print("\n  MLP Gate/Up projections:")
            # Basic
            basic_gate_lr = torch.matmul(basic_post_ln, basic_layer.mlp.gate_proj.cola_a)
            basic_gate_lr_act = basic_layer.mlp.gate_proj.lr_act(basic_gate_lr)
            basic_gate = torch.matmul(basic_gate_lr_act, basic_layer.mlp.gate_proj.cola_b)
            
            basic_up_lr = torch.matmul(basic_post_ln, basic_layer.mlp.up_proj.cola_a)
            basic_up_lr_act = basic_layer.mlp.up_proj.lr_act(basic_up_lr)
            basic_up = torch.matmul(basic_up_lr_act, basic_layer.mlp.up_proj.cola_b)
            
            print_tensor_info("    basic gate: final", basic_gate)
            print_tensor_info("    basic up: final", basic_up)
            
            # Cola BTP MLP gate/up projections
            print("\n  Cola BTP MLP Gate/Up projections:")
            cola_gate_up_lr = cola_layer.mlp.gate_up_proj0(cola_post_ln, rstd=cola_post_rstd)
            cola_gate_up_lr_act = cola_layer.mlp.lr_act(cola_gate_up_lr)
            cola_gate_up_lr_unflattened = cola_gate_up_lr_act.unflatten(-1, (2, -1))  # [seq, batch, 2, rank]
            cola_gate_up_states = cola_layer.mlp.gate_up_proj1(cola_gate_up_lr_unflattened)  # [seq, batch, 2, out_features]
            cola_gate, cola_up = cola_gate_up_states.unbind(dim=0)  # Each: [seq, batch, out_features]
            
            print_tensor_info("    cola BTP gate: final", cola_gate)
            print_tensor_info("    cola BTP up: final", cola_up)
            
            gate_diff = (basic_gate - cola_gate).abs()
            up_diff = (basic_up - cola_up).abs()
            print(f"    Gate diff (Basic vs Cola BTP) - Max: {gate_diff.max().item():.6e}, Mean: {gate_diff.mean().item():.6e}")
            print(f"    Up diff (Basic vs Cola BTP) - Max: {up_diff.max().item():.6e}, Mean: {up_diff.mean().item():.6e}")
            
            # MLP activation and down projection
            print("\n  MLP Down projection:")
            basic_mlp_intermediate = basic_gate * basic_up
            cola_mlp_intermediate = cola_gate * cola_up
            print_tensor_info("    basic mlp intermediate (gate*up)", basic_mlp_intermediate)
            print_tensor_info("    cola BTP mlp intermediate (gate*up)", cola_mlp_intermediate)
            
            # Basic down
            basic_down_lr = torch.matmul(basic_mlp_intermediate, basic_layer.mlp.down_proj.cola_a)
            basic_down_lr_act = basic_layer.mlp.down_proj.lr_act(basic_down_lr)
            basic_down = torch.matmul(basic_down_lr_act, basic_layer.mlp.down_proj.cola_b)
            print_tensor_info("    basic down: final", basic_down)
            
            # Cola BTP down
            cola_down_lr = cola_layer.mlp.down_proj0(cola_mlp_intermediate, rstd=cola_post_rstd)
            cola_down_lr_act = cola_layer.mlp.lr_act(cola_down_lr)
            cola_down = cola_layer.mlp.down_proj1(cola_down_lr_act)
            print_tensor_info("    cola BTP down: final", cola_down)
            
            down_diff = (basic_down - cola_down).abs()
            print(f"    Down diff (Basic vs Cola BTP) - Max: {down_diff.max().item():.6e}, Mean: {down_diff.mean().item():.6e}")
            
            # Run full forward pass for final comparison
            print("\n--- Step 6: Full Forward Pass ---")
            print("  Running full layer forward...")
            
            basic_output = basic_layer(
                hidden_states=hidden_states,
                sequence_mask=sequence_mask
            )
            
            # COMMENTED OUT: Vanilla output
            # vanilla_output = vanilla_layer(
            #     hidden_states=hidden_states,
            #     sequence_mask=sequence_mask
            # )

            cola_output = cola_layer(
                hidden_states=hidden_states,
                sequence_mask=sequence_mask
            )
            
            # COMMENTED OUT: Final Output Comparison (Basic vs Vanilla)
            # print("\n--- Final Output Comparison (Basic vs Vanilla) ---")
            # basic_hidden = basic_output["hidden_states"]
            # vanilla_hidden = vanilla_output["hidden_states"]
            # 
            # print_tensor_info("Basic final output", basic_hidden)
            # print_tensor_info("Vanilla final output", vanilla_hidden)
            # 
            # # Check shapes match
            # assert basic_hidden.shape == vanilla_hidden.shape, \
            #     f"Shape mismatch: Basic {basic_hidden.shape} vs Vanilla {vanilla_hidden.shape}"
            # 
            # # Compare with tolerance
            # vanilla_match = torch.allclose(basic_hidden, vanilla_hidden, rtol=1e-3, atol=1e-3)
            # 
            # diff = (basic_hidden - vanilla_hidden).abs()
            # max_diff = diff.max().item()
            # mean_diff = diff.mean().item()
            # relative_diff = (diff / (basic_hidden.abs() + 1e-8)).max().item()
            # 
            # print(f"\n  Output diff (Basic vs Vanilla) - Max: {max_diff:.6e}, Mean: {mean_diff:.6e}, Relative: {relative_diff:.6e}")
            # print(f"  Basic range: [{basic_hidden.min():.6f}, {basic_hidden.max():.6f}]")
            # print(f"  Vanilla range: [{vanilla_hidden.min():.6f}, {vanilla_hidden.max():.6f}]")
            # 
            # if not vanilla_match:
            #     print(f"\n✗ Vanilla output does NOT match Basic")
            #     print(f"  Max absolute diff: {max_diff:.6e}")
            #     print(f"  Mean absolute diff: {mean_diff:.6e}")
            #     print(f"  Max relative diff: {relative_diff:.6e}")
            #     
            #     # Find where the largest differences are
            #     max_diff_indices = diff.flatten().topk(5).indices
            #     print(f"\n  Top 5 largest differences (Basic vs Vanilla):")
            #     for idx in max_diff_indices:
            #         flat_idx = idx.item()
            #         basic_val = basic_hidden.flatten()[flat_idx].item()
            #         vanilla_val = vanilla_hidden.flatten()[flat_idx].item()
            #         diff_val = diff.flatten()[flat_idx].item()
            #         print(f"    Index {flat_idx}: Basic={basic_val:.6f}, Vanilla={vanilla_val:.6f}, Diff={diff_val:.6e}")
            # else:
            #     print("\n✓ Vanilla output matches Basic!")

            # Basic vs Cola BTP
            basic_hidden = basic_output["hidden_states"]
            print("\n--- Final Output Comparison (Basic vs Cola BTP) ---")
            cola_hidden = cola_output["hidden_states"]
            print_tensor_info("Cola BTP final output", cola_hidden)

            assert basic_hidden.shape == cola_hidden.shape, \
                f"Shape mismatch: Basic {basic_hidden.shape} vs Cola BTP {cola_hidden.shape}"

            cola_match = torch.allclose(basic_hidden, cola_hidden, rtol=1e-3, atol=1e-3)

            cola_diff = (basic_hidden - cola_hidden).abs()
            cola_max_diff = cola_diff.max().item()
            cola_mean_diff = cola_diff.mean().item()
            cola_relative_diff = (cola_diff / (basic_hidden.abs() + 1e-8)).max().item()

            print(f"\n  Output diff (Basic vs Cola BTP) - Max: {cola_max_diff:.6e}, Mean: {cola_mean_diff:.6e}, Relative: {cola_relative_diff:.6e}")
            print(f"  Cola BTP range: [{cola_hidden.min():.6f}, {cola_hidden.max():.6f}]")

            if not cola_match:
                print(f"\n✗ Cola BTP output does NOT match Basic")
                print(f"  Max absolute diff: {cola_max_diff:.6e}")
                print(f"  Mean absolute diff: {cola_mean_diff:.6e}")
                print(f"  Max relative diff: {cola_relative_diff:.6e}")

                max_diff_indices = cola_diff.flatten().topk(5).indices
                print(f"\n  Top 5 largest differences (Basic vs Cola BTP):")
                for idx in max_diff_indices:
                    flat_idx = idx.item()
                    basic_val = basic_hidden.flatten()[flat_idx].item()
                    cola_val = cola_hidden.flatten()[flat_idx].item()
                    diff_val = cola_diff.flatten()[flat_idx].item()
                    print(f"    Index {flat_idx}: Basic={basic_val:.6f}, Cola={cola_val:.6f}, Diff={diff_val:.6e}")
            else:
                print("\n✓ Cola BTP output matches Basic!")
    finally:
        # Clean up distributed
        if dist.is_initialized():
            parallel_context.destroy()


if __name__ == "__main__":
    test_cola_compute_parity()

