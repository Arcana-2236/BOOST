#!/usr/bin/env python3
"""
Test script to compare weight initialization between:
- basic_cola_llama.py (reference implementation)
- vanilla_cola_llama.py (vanilla TP implementation)
- cola_llama.py (BTP implementation)

This script:
1. Initializes all three models with the same config and seed
2. Compares weight initialization values for each layer
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict

# Add the examples/cola directory to path for imports
cola_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cola_dir)

from config_basic_cola_llama import BasicColaLlamaConfig
from config_cola_llama import ColaLlamaConfig
from basic_cola_llama import BasicColaLlamaForTraining
from vanilla_cola_llama import VanillaColaLlamaForTraining
from cola_llama import ColaLlamaForTraining
from nanotron.config import ParallelismArgs, Config, GeneralArgs, ModelArgs, TokenizerArgs
from nanotron.config.config import (
    CheckpointsArgs, LoggingArgs, TokensArgs, OptimizerArgs, LRSchedulerArgs, AdamWOptimizerArgs,
    DatasetStageArgs, DataArgs, PretrainDatasetsArgs
)
from nanotron.config.models_config import RandomInit
from nanotron.random import set_random_seed
from nanotron import distributed as dist
from nanotron.trainer import DistributedTrainer
from pathlib import Path


def create_test_config():
    """Create a small test config for faster testing"""
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
    )
    
    cola_config = ColaLlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        vocab_size=1000,
        max_position_embeddings=512,
        attn_rank=64,
        mlp_rank=64,
    )
    
    return basic_config, vanilla_config, cola_config


def create_full_config(model_config, seed=42):
    """Create a complete Config object"""
    parallel_config = ParallelismArgs(dp=1, tp=1, pp=1, expert_parallel_size=1)
    
    config = Config(
        general=GeneralArgs(project="test", run="test", seed=seed),
        parallelism=parallel_config,
        model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
        tokenizer=TokenizerArgs(),
        checkpoints=CheckpointsArgs(
            checkpoints_path=Path("/tmp/test_checkpoints"),
            checkpoint_interval=100,
            save_initial_state=False,
            save_final_state=False,
        ),
        logging=LoggingArgs(
            log_level="info",
            log_level_replica="info",
            iteration_step_info_interval=1,
        ),
        optimizer=OptimizerArgs(
            zero_stage=0,
            weight_decay=0.01,
            clip_grad=1.0,
            accumulate_grad_in_fp32=False,
            optimizer_factory=AdamWOptimizerArgs(
                name="adamW",
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-8,
                torch_adam_is_fused=True,
            ),
            learning_rate_scheduler=LRSchedulerArgs(
                learning_rate=1e-4,
                lr_warmup_steps=10,
                lr_warmup_style="linear",
                lr_decay_style="cosine",
                min_decay_lr=1e-5,
            ),
        ),
        tokens=TokensArgs(
            sequence_length=32,
            micro_batch_size=2,
            batch_accumulation_per_replica=1,
            train_steps=10,
            limit_test_batches=0,
            limit_val_batches=0,
            val_check_interval=-1,
        ),
        data_stages=[
            DatasetStageArgs(
                name="test_stage",
                start_training_step=1,
                data=DataArgs(
                    dataset=PretrainDatasetsArgs(
                        hf_dataset_or_datasets="dummy",
                        hf_dataset_splits="train",
                        text_column_name="text",
                    ),
                    num_loading_workers=1,
                    seed=42,
                ),
            )
        ],
    )
    
    return config


def get_weight_dict(model, prefix=""):
    """Extract all weights from a model as a dictionary"""
    weights = OrderedDict()
    for name, param in model.named_parameters():
        full_name = f"{prefix}.{name}" if prefix else name
        weights[full_name] = param.data.clone()
    return weights


def map_basic_to_other_weight_name(basic_name: str, is_vanilla: bool = True):
    """
    Map basic_cola_llama.py weight names to vanilla_cola_llama.py or cola_llama.py names.
    
    Basic uses: cola_a, cola_b
    Vanilla/Cola use: proj0, proj1
    
    Basic structure: model.decoder.{layer}.pp_block.{attn|mlp}.{q|k|v|o|gate|up|down}_proj.{cola_a|cola_b}
    Vanilla/Cola structure: model.decoder.{layer}.pp_block.{attn|mlp}.{qkv|gate_up|down}_proj{0|1}.weight
    
    Also handles grouped GEMMs in cola_llama.py (qkv_proj0, gate_up_proj0, etc.)
    """
    # Remove prefix if present
    if basic_name.startswith("basic."):
        basic_name = basic_name[6:]
    
    # Check if it's a CoLA layer (cola_a or cola_b)
    if ".cola_a" in basic_name or ".cola_b" in basic_name:
        # Determine if it's cola_a (proj0) or cola_b (proj1)
        is_cola_a = ".cola_a" in basic_name
        proj_type = "proj0" if is_cola_a else "proj1"
        
        # Extract layer info
        parts = basic_name.split(".")
        # Find the projection name (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        proj_name = None
        for part in parts:
            if part.endswith("_proj") and part != "proj":
                proj_name = part
                break
        
        if proj_name is None:
            return None
        
        # Map to vanilla/cola structure
        # Basic: model.decoder.0.pp_block.attn.q_proj.cola_a
        # Vanilla/Cola: model.decoder.0.pp_block.attn.qkv_proj0.weight (for qkv) or q_proj1.weight (for q)
        
        # Find layer index and module type (attn or mlp)
        layer_idx = None
        module_type = None
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0 and parts[i-1] == "decoder":
                layer_idx = part
            if part in ["attn", "mlp"]:
                module_type = part
        
        if layer_idx is None or module_type is None:
            return None
        
        # Handle grouped vs separate projections
        if is_vanilla:
            # Vanilla: separate projections (q_proj1, k_proj1, v_proj1, gate_proj1, up_proj1)
            # proj0 is ColumnLinear (down), proj1 is RowLinear (up)
            if proj_name in ["q_proj", "k_proj", "v_proj"]:
                # For attention: qkv_proj0 is ColumnLinear, q_proj1/k_proj1/v_proj1 are RowLinear
                if is_cola_a:  # cola_a -> proj0 (ColumnLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.qkv_proj0.weight"
                else:  # cola_b -> proj1 (RowLinear)
                    if proj_name == "q_proj":
                        return f"model.decoder.{layer_idx}.pp_block.{module_type}.q_proj1.weight"
                    elif proj_name == "k_proj":
                        return f"model.decoder.{layer_idx}.pp_block.{module_type}.k_proj1.weight"
                    elif proj_name == "v_proj":
                        return f"model.decoder.{layer_idx}.pp_block.{module_type}.v_proj1.weight"
            elif proj_name in ["gate_proj", "up_proj"]:
                # For MLP: gate_up_proj0 is ColumnLinear, gate_proj1/up_proj1 are RowLinear
                if is_cola_a:  # cola_a -> proj0 (ColumnLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_up_proj0.weight"
                else:  # cola_b -> proj1 (RowLinear)
                    if proj_name == "gate_proj":
                        return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_proj1.weight"
                    elif proj_name == "up_proj":
                        return f"model.decoder.{layer_idx}.pp_block.{module_type}.up_proj1.weight"
            elif proj_name == "o_proj":
                # o_proj: o_proj0 is ColumnLinear, o_proj1 is RowLinear
                if is_cola_a:  # cola_a -> proj0 (ColumnLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj0.weight"
                else:  # cola_b -> proj1 (RowLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj1.weight"
            elif proj_name == "down_proj":
                # down_proj: down_proj0 is ColumnLinear, down_proj1 is RowLinear
                if is_cola_a:  # cola_a -> proj0 (ColumnLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj0.weight"
                else:  # cola_b -> proj1 (RowLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj1.weight"
        else:
            # Cola (BTP): grouped projections
            # proj0 is RowLinear (down), proj1 is ColumnLinear/BatchedColumnLinear (up)
            if proj_name in ["q_proj", "k_proj", "v_proj"]:
                # For attention: qkv_proj0 is RowLinear (grouped), qkv_proj1 is BatchedColumnLinear
                if is_cola_a:  # cola_a -> proj0 (RowLinear, grouped)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.qkv_proj0.weight"
                else:  # cola_b -> proj1 (BatchedColumnLinear, grouped)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.qkv_proj1.weight"
            elif proj_name in ["gate_proj", "up_proj"]:
                # For MLP: gate_up_proj0 is RowLinear (grouped), gate_up_proj1 is BatchedColumnLinear
                if is_cola_a:  # cola_a -> proj0 (RowLinear, grouped)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_up_proj0.weight"
                else:  # cola_b -> proj1 (BatchedColumnLinear, grouped)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_up_proj1.weight"
            elif proj_name == "o_proj":
                # o_proj: o_proj0 is RowLinear, o_proj1 is ColumnLinear
                if is_cola_a:  # cola_a -> proj0 (RowLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj0.weight"
                else:  # cola_b -> proj1 (ColumnLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj1.weight"
            elif proj_name == "down_proj":
                # down_proj: down_proj0 is RowLinear, down_proj1 is ColumnLinear
                if is_cola_a:  # cola_a -> proj0 (RowLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj0.weight"
                else:  # cola_b -> proj1 (ColumnLinear)
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj1.weight"
    
    return None


def extract_cola_weight_from_grouped(weight_tensor: torch.Tensor, proj_name: str, proj_idx: int, is_proj0: bool):
    """
    Extract individual CoLA weight from grouped GEMM weight tensor.
    
    For grouped GEMMs:
    - qkv_proj0: [hidden_size, 3*attn_rank] -> extract q/k/v at indices [:, 0:rank], [:, rank:2*rank], [:, 2*rank:3*rank]
    - gate_up_proj0: [hidden_size, 2*mlp_rank] -> extract gate/up at indices [:, 0:rank], [:, rank:2*rank]
    - qkv_proj1: [3, attn_rank, out_features] -> extract q/k/v at indices [0, :, :], [1, :, :], [2, :, :]
    - gate_up_proj1: [2, mlp_rank, out_features] -> extract gate/up at indices [0, :, :], [1, :, :]
    """
    if is_proj0:
        # proj0: grouped along output dimension
        # PyTorch Linear stores as [out_features, in_features]
        # So qkv_proj0.weight is [3*rank, hidden_size] (not [hidden_size, 3*rank])
        if "qkv" in proj_name:
            # qkv_proj0: [3*rank, hidden_size] (Linear format)
            rank = weight_tensor.shape[0] // 3
            if proj_idx == 0:  # q
                return weight_tensor[0:rank, :].t()  # Extract q slice and transpose to [hidden_size, rank]
            elif proj_idx == 1:  # k
                return weight_tensor[rank:2*rank, :].t()  # Extract k slice and transpose
            elif proj_idx == 2:  # v
                return weight_tensor[2*rank:3*rank, :].t()  # Extract v slice and transpose
        elif "gate_up" in proj_name or "gate" in proj_name or "up" in proj_name:
            # gate_up_proj0: [2*rank, hidden_size] (Linear format)
            rank = weight_tensor.shape[0] // 2
            if proj_idx == 0:  # gate
                return weight_tensor[0:rank, :].t()  # Extract gate slice and transpose to [hidden_size, rank]
            elif proj_idx == 1:  # up
                return weight_tensor[rank:2*rank, :].t()  # Extract up slice and transpose
    else:
        # proj1: grouped along batch dimension (first dim)
        # For BatchedTensorParallelColumnLinear, the shape is [batch, rank, out_features]
        # But PyTorch Linear stores as [out_features, in_features], so it might be [batch, out_features, rank]
        # Let's check the shape to determine the format
        if len(weight_tensor.shape) == 3:
            # 3D tensor: [batch, ...]
            if "qkv" in proj_name:
                # qkv_proj1: [3, out_features, rank] or [3, rank, out_features]
                # Check which dimension is rank (should be smaller)
                if weight_tensor.shape[1] < weight_tensor.shape[2]:
                    # [3, rank, out_features] - transpose to [rank, out_features]
                    if proj_idx == 0:  # q
                        return weight_tensor[0, :, :].t()  # [rank, out_features] -> [out_features, rank] -> transpose to [rank, out_features]
                    elif proj_idx == 1:  # k
                        return weight_tensor[1, :, :].t()
                    elif proj_idx == 2:  # v
                        return weight_tensor[2, :, :].t()
                else:
                    # [3, out_features, rank] - already in correct format
                    if proj_idx == 0:  # q
                        return weight_tensor[0, :, :].t()  # [out_features, rank] -> transpose to [rank, out_features]
                    elif proj_idx == 1:  # k
                        return weight_tensor[1, :, :].t()
                    elif proj_idx == 2:  # v
                        return weight_tensor[2, :, :].t()
            elif "gate_up" in proj_name or "gate" in proj_name or "up" in proj_name:
                # gate_up_proj1: [2, out_features, rank] or [2, rank, out_features]
                if weight_tensor.shape[1] < weight_tensor.shape[2]:
                    # [2, rank, out_features]
                    if proj_idx == 0:  # gate
                        return weight_tensor[0, :, :].t()
                    elif proj_idx == 1:  # up
                        return weight_tensor[1, :, :].t()
                else:
                    # [2, out_features, rank]
                    if proj_idx == 0:  # gate
                        return weight_tensor[0, :, :].t()
                    elif proj_idx == 1:  # up
                        return weight_tensor[1, :, :].t()
    
    return weight_tensor


def compare_equivalent_weights(basic_weights: Dict, other_weights: Dict, name1: str, name2: str, is_vanilla: bool = True, rtol=1e-3, atol=1e-5):
    """Compare weights between basic and vanilla/cola, handling different naming and grouped GEMMs"""
    results = []
    
    # Determine the prefix for other_weights (vanilla or cola)
    other_prefix = "vanilla" if is_vanilla else "cola"
    
    # Process all basic weights
    for basic_name, basic_weight in basic_weights.items():
        if ".cola_a" not in basic_name and ".cola_b" not in basic_name:
            # Not a CoLA weight, skip or handle separately
            continue
        
        # Map basic name to other name (without prefix)
        other_name_no_prefix = map_basic_to_other_weight_name(basic_name, is_vanilla=is_vanilla)
        
        if other_name_no_prefix is None:
            basic_std = basic_weight.std().item()
            results.append((basic_name, f"Could not map to {name2}", None, None, None, basic_std, None))
            continue
        
        # Add the prefix to match the actual weight dict keys
        other_name = f"{other_prefix}.{other_name_no_prefix}"
        
        if other_name not in other_weights:
            # Try to find the actual key in other_weights (search for partial match)
            # The actual key might have a different structure
            matching_keys = [k for k in other_weights.keys() if other_name_no_prefix in k or k.endswith(other_name_no_prefix)]
            if matching_keys:
                # Prefer exact match after prefix
                exact_match = [k for k in matching_keys if k == other_name]
                if exact_match:
                    other_name = exact_match[0]
                else:
                    other_name = matching_keys[0]  # Use the first matching key
            else:
                basic_std = basic_weight.std().item()
                results.append((basic_name, f"Mapped to {other_name} but missing in {name2}", None, None, other_name, basic_std, None))
                continue
        
        other_weight = other_weights[other_name]
        
        # Extract individual weight from grouped GEMM if needed
        # Mapping: cola_a -> proj0, cola_b -> proj1
        is_cola_a = ".cola_a" in basic_name
        is_cola_b = ".cola_b" in basic_name
        
        # Determine which projection we're comparing and if it's grouped
        proj_name = None
        proj_idx = None
        needs_extraction = False
        
        if "q_proj" in basic_name:
            proj_idx = 0  # q is first in qkv
            if is_cola_a:
                # cola_a -> proj0, which is grouped (qkv_proj0 or gate_up_proj0)
                proj_name = "qkv"
                needs_extraction = True
            else:  # is_cola_b
                # cola_b -> proj1
                # For vanilla: q_proj1 is separate (not grouped)
                # For cola: qkv_proj1 is grouped
                if is_vanilla:
                    needs_extraction = False  # Separate layer
                else:
                    proj_name = "qkv"
                    needs_extraction = True  # Grouped in cola
        elif "k_proj" in basic_name:
            proj_idx = 1  # k is second in qkv
            if is_cola_a:
                proj_name = "qkv"
                needs_extraction = True
            else:  # is_cola_b
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "qkv"
                    needs_extraction = True
        elif "v_proj" in basic_name:
            proj_idx = 2  # v is third in qkv
            if is_cola_a:
                proj_name = "qkv"
                needs_extraction = True
            else:  # is_cola_b
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "qkv"
                    needs_extraction = True
        elif "gate_proj" in basic_name:
            proj_idx = 0  # gate is first in gate_up
            if is_cola_a:
                proj_name = "gate_up"
                needs_extraction = True
            else:  # is_cola_b
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "gate_up"
                    needs_extraction = True
        elif "up_proj" in basic_name:
            proj_idx = 1  # up is second in gate_up
            if is_cola_a:
                proj_name = "gate_up"
                needs_extraction = True
            else:  # is_cola_b
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "gate_up"
                    needs_extraction = True
        
        # Extract from grouped tensor if needed
        if needs_extraction and proj_name and ("qkv" in proj_name or "gate_up" in proj_name):
            # Extract the individual weight slice from the grouped tensor
            other_weight = extract_cola_weight_from_grouped(other_weight, proj_name, proj_idx, is_cola_a)
        
        # Handle transpose: PyTorch Linear stores weights as [out_features, in_features]
        # Basic CoLA: cola_a is [in_features, rank], cola_b is [rank, out_features]
        # After extraction from grouped tensors, we've already transposed proj0
        # For non-grouped layers (o_proj0, down_proj0, and proj1), we need to transpose
        
        # Check if shapes are transposed and fix
        # This handles: o_proj0, down_proj0, and proj1 layers that aren't grouped
        if basic_weight.shape == tuple(reversed(other_weight.shape)):
            other_weight = other_weight.t()
        
        # Compare shapes
        if basic_weight.shape != other_weight.shape:
            # Calculate std even for shape mismatches
            basic_std = basic_weight.std().item()
            other_std = other_weight.std().item()
            results.append((
                basic_name,
                f"Shape mismatch: {basic_weight.shape} vs {other_weight.shape}",
                None, None, other_name, basic_std, other_std
            ))
            continue
        
        # Calculate statistics
        diff = basic_weight - other_weight
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        std_diff = diff.abs().std().item()
        is_close = torch.allclose(basic_weight, other_weight, rtol=rtol, atol=atol)
        
        # Calculate std of each weight tensor
        basic_std = basic_weight.std().item()
        other_std = other_weight.std().item()
        
        results.append((
            basic_name,
            "MATCH" if is_close else "DIFF",
            max_diff,
            mean_diff,
            other_name,
            basic_std,
            other_std
        ))
    
    return results


def compare_weights(weights1: Dict, weights2: Dict, name1: str, name2: str, rtol=1e-3, atol=1e-5):
    """Compare two weight dictionaries (simple version for non-CoLA weights)"""
    all_keys = set(weights1.keys()) | set(weights2.keys())
    results = []
    
    for key in sorted(all_keys):
        if key not in weights1:
            results.append((key, f"Missing in {name1}", None, None, None))
            continue
        if key not in weights2:
            results.append((key, f"Missing in {name2}", None, None, None))
            continue
        
        w1 = weights1[key]
        w2 = weights2[key]
        
        if w1.shape != w2.shape:
            results.append((key, f"Shape mismatch: {w1.shape} vs {w2.shape}", None, None, None))
            continue
        
        # Calculate statistics
        diff = w1 - w2
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        std_diff = diff.abs().std().item()
        is_close = torch.allclose(w1, w2, rtol=rtol, atol=atol)
        
        results.append((
            key,
            "MATCH" if is_close else "DIFF",
            max_diff,
            mean_diff,
            None
        ))
    
    return results


def print_weight_comparison(results, name1: str, name2: str):
    """Print weight comparison results"""
    print(f"\n{'='*80}")
    print(f"Weight Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    
    matches = sum(1 for _, status, _, _, _, _, _ in results if status == "MATCH")
    diffs = sum(1 for _, status, _, _, _, _, _ in results if status == "DIFF")
    missing = sum(1 for _, status, _, _, _, _, _ in results if status and ("Missing" in status or "Could not map" in status or "missing in" in status.lower() or "Shape mismatch" in status))
    
    print(f"Summary: {matches} matches, {diffs} differences, {missing} missing/unmapped/shape_mismatch")
    print(f"\n{'Basic Layer':<50} {'Status':<10} {'Max Diff':<15} {'Mean Diff':<15} {'Basic Std':<15} {'Other Std':<15} {'Mapped To':<50}")
    print("-" * 170)
    
    for key, status, max_diff, mean_diff, mapped_to, basic_std, other_std in results:
        max_str = f"{max_diff:.6e}" if max_diff is not None else "N/A"
        mean_str = f"{mean_diff:.6e}" if mean_diff is not None else "N/A"
        basic_std_str = f"{basic_std:.6e}" if basic_std is not None else "N/A"
        other_std_str = f"{other_std:.6e}" if other_std is not None else "N/A"
        mapped_str = mapped_to if mapped_to else "N/A"
        print(f"{key:<50} {status:<10} {max_str:<15} {mean_str:<15} {basic_std_str:<15} {other_std_str:<15} {mapped_str:<50}")




def main():
    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    # Set random seed for reproducibility
    seed = 42
    set_random_seed(seed)
    torch.manual_seed(seed)
    
    # Create configs
    basic_config, vanilla_config, cola_config = create_test_config()
    
    # Create full configs
    basic_full_config = create_full_config(basic_config, seed=seed)
    vanilla_full_config = create_full_config(vanilla_config, seed=seed)
    cola_full_config = create_full_config(cola_config, seed=seed)
    
    # Initialize models using DistributedTrainer (like train_cola.py)
    print("Initializing models...")
    
    # Basic model
    set_random_seed(seed)
    torch.manual_seed(seed)
    basic_trainer = DistributedTrainer(
        config_or_config_file=basic_full_config,
        model_config_class=basic_config.__class__,
        model_class=BasicColaLlamaForTraining,
    )
    basic_model = basic_trainer.unwrapped_model
    
    # Vanilla model
    set_random_seed(seed)
    torch.manual_seed(seed)
    vanilla_trainer = DistributedTrainer(
        config_or_config_file=vanilla_full_config,
        model_config_class=vanilla_config.__class__,
        model_class=VanillaColaLlamaForTraining,
    )
    vanilla_model = vanilla_trainer.unwrapped_model
    
    # Cola model
    set_random_seed(seed)
    torch.manual_seed(seed)
    cola_trainer = DistributedTrainer(
        config_or_config_file=cola_full_config,
        model_config_class=cola_config.__class__,
        model_class=ColaLlamaForTraining,
    )
    cola_model = cola_trainer.unwrapped_model
    
    print("Models initialized!")
    
    # Extract weights
    print("\n" + "="*80)
    print("COMPARING WEIGHT INITIALIZATION")
    print("="*80)
    
    basic_weights = get_weight_dict(basic_model, "basic")
    vanilla_weights = get_weight_dict(vanilla_model, "vanilla")
    cola_weights = get_weight_dict(cola_model, "cola")
    
    # Debug: Print some sample weight names to understand the structure
    if dist.get_rank() == 0:
        print("\nSample Basic weight names:")
        for name in list(basic_weights.keys())[:5]:
            print(f"  {name}")
        print("\nSample Vanilla weight names:")
        for name in list(vanilla_weights.keys())[:5]:
            print(f"  {name}")
        print("\nSample Cola weight names:")
        for name in list(cola_weights.keys())[:5]:
            print(f"  {name}")
    
    # Compare weights (using equivalent weight mapping for CoLA layers)
    print("\n1. Basic vs Vanilla (CoLA weights):")
    basic_vs_vanilla = compare_equivalent_weights(basic_weights, vanilla_weights, "Basic", "Vanilla", is_vanilla=True)
    print_weight_comparison(basic_vs_vanilla, "Basic", "Vanilla")
    
    print("\n2. Basic vs Cola (CoLA weights):")
    basic_vs_cola = compare_equivalent_weights(basic_weights, cola_weights, "Basic", "Cola", is_vanilla=False)
    print_weight_comparison(basic_vs_cola, "Basic", "Cola")
    
    print("\n" + "="*80)
    print("INITIALIZATION METHOD SUMMARY")
    print("="*80)
    print("\nBasic Cola Llama:")
    print("  - CoLA weights initialized in ColaLayer.__init__ (during model creation)")
    print("  - Uses: torch.randn(...) / rank**(1/4) * target_sdv**(1/2)")
    print("  - Formula: target_sdv = (in_features + out_features)**(-1/2)")
    print("  - Effective std = rank**(-1/4) * target_sdv**(1/2)")
    print("  - init_model_randomly() SKIPS ColaLayer parameters (line 1214-1217)")
    print("\nVanilla/Cola Llama:")
    print("  - CoLA weights initialized in init_model_randomly() using StandardParametrizator")
    print("  - Uses: init.normal_(weight, mean=0.0, std=cola_std)")
    print("  - cola_std calculated by get_cola_std()")
    print("  - For RowLinear: compensates by multiplying cola_std by sqrt(2 * num_layers)")
    print("  - StandardParametrizator then divides by sqrt(2 * num_layers) for RowLinear")
    print("\nNOTE: Even with same seed, weights may differ because:")
    print("  - Basic initializes during __init__ (uses RNG state at model creation)")
    print("  - Vanilla/Cola initialize during init_model_randomly (uses RNG state later)")
    print("  - Different RNG consumption between model creation and initialization")
    print("  - Other parameters (embeddings, layer norms) are initialized in between")
    print("\n" + "="*80)
    print("WEIGHT INITIALIZATION COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

