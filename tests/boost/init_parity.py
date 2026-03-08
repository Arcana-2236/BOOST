#!/usr/bin/env python3
"""
Minimal test script to check initialization parity between:
- Basic Cola Llama (reference implementation)
- Vanilla Cola Llama (vanilla TP)
- Cola Llama (BTP)

This script ONLY checks weight initialization, no forward pass.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from pathlib import Path
from collections import OrderedDict

# Force deterministic behavior
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Imports from the same directory
from config_basic_cola_llama import BasicColaLlamaConfig
from config_cola_llama import ColaLlamaConfig
from basic_cola_llama import BasicColaLlamaForTraining
from vanilla_cola_llama import VanillaColaLlamaForTraining
from cola_llama import ColaLlamaForTraining

from nanotron.config import Config, ParallelismArgs, GeneralArgs, ModelArgs, TokenizerArgs
from nanotron.config.config import (
    CheckpointsArgs, LoggingArgs, TokensArgs, OptimizerArgs, LRSchedulerArgs, AdamWOptimizerArgs,
    DatasetStageArgs, DataArgs, PretrainDatasetsArgs
)
from nanotron.config.models_config import RandomInit
from nanotron.random import set_random_seed
from nanotron import distributed as dist
from nanotron.trainer import DistributedTrainer


def create_test_config(model_config, seed=42):
    """Create a minimal Config for initialization testing"""
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


def get_param_dict(model, prefix=""):
    """Extract all parameters from a model as a dictionary"""
    params = OrderedDict()
    for name, param in model.named_parameters():
        full_name = f"{prefix}.{name}" if prefix else name
        params[full_name] = param.data.detach().cpu().clone()
    return params


def map_basic_to_other_weight_name(basic_name: str, is_vanilla: bool = True) -> Optional[str]:
    """
    Map basic_cola_llama.py weight names to vanilla_cola_llama.py or cola_llama.py names.
    
    Basic uses: cola_a, cola_b
    Vanilla/Cola use: proj0, proj1
    """
    # Remove prefix if present
    if basic_name.startswith("basic."):
        basic_name = basic_name[6:]
    
    # Check if it's a CoLA layer (cola_a or cola_b)
    if ".cola_a" not in basic_name and ".cola_b" not in basic_name:
        return None
    
    # Determine if it's cola_a (proj0) or cola_b (proj1)
    is_cola_a = ".cola_a" in basic_name
    
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
            if is_cola_a:  # cola_a -> proj0 (ColumnLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_up_proj0.weight"
            else:  # cola_b -> proj1 (RowLinear)
                if proj_name == "gate_proj":
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_proj1.weight"
                elif proj_name == "up_proj":
                    return f"model.decoder.{layer_idx}.pp_block.{module_type}.up_proj1.weight"
        elif proj_name == "o_proj":
            if is_cola_a:  # cola_a -> proj0 (ColumnLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj0.weight"
            else:  # cola_b -> proj1 (RowLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj1.weight"
        elif proj_name == "down_proj":
            if is_cola_a:  # cola_a -> proj0 (ColumnLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj0.weight"
            else:  # cola_b -> proj1 (RowLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj1.weight"
    else:
        # Cola (BTP): grouped projections
        # proj0 is RowLinear (down), proj1 is ColumnLinear/BatchedColumnLinear (up)
        if proj_name in ["q_proj", "k_proj", "v_proj"]:
            if is_cola_a:  # cola_a -> proj0 (RowLinear, grouped)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.qkv_proj0.weight"
            else:  # cola_b -> proj1 (BatchedColumnLinear, grouped)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.qkv_proj1.weight"
        elif proj_name in ["gate_proj", "up_proj"]:
            if is_cola_a:  # cola_a -> proj0 (RowLinear, grouped)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_up_proj0.weight"
            else:  # cola_b -> proj1 (BatchedColumnLinear, grouped)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.gate_up_proj1.weight"
        elif proj_name == "o_proj":
            if is_cola_a:  # cola_a -> proj0 (RowLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj0.weight"
            else:  # cola_b -> proj1 (ColumnLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.o_proj1.weight"
        elif proj_name == "down_proj":
            if is_cola_a:  # cola_a -> proj0 (RowLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj0.weight"
            else:  # cola_b -> proj1 (ColumnLinear)
                return f"model.decoder.{layer_idx}.pp_block.{module_type}.down_proj1.weight"
    
    return None


def extract_cola_weight_from_grouped(weight_tensor: torch.Tensor, proj_name: str, proj_idx: int, is_proj0: bool) -> torch.Tensor:
    """
    Extract individual CoLA weight from grouped GEMM weight tensor.
    
    DETAILED GROUPING EXPLANATION:
    ===============================
    
    Basic Model (no grouping):
    - q_proj.cola_a: [hidden_size, rank]  (separate tensor)
    - k_proj.cola_a: [hidden_size, rank]  (separate tensor)
    - v_proj.cola_a: [hidden_size, rank]  (separate tensor)
    
    Vanilla/Cola Model (with grouping):
    - qkv_proj0: TensorParallelRowLinear(hidden_size, 3*rank)
      * PyTorch Linear stores as [out_features, in_features] = [3*rank, hidden_size]
      * Contains q, k, v stacked along output dimension (first dim)
      * q: rows [0:rank, :] -> shape [rank, hidden_size]
      * k: rows [rank:2*rank, :] -> shape [rank, hidden_size]
      * v: rows [2*rank:3*rank, :] -> shape [rank, hidden_size]
      * After extraction: [rank, hidden_size], then transpose to [hidden_size, rank] to match Basic
    
    For proj0 (cola_a equivalent):
    - qkv_proj0: [3*rank, hidden_size] -> extract q/k/v at indices 0/1/2
    - gate_up_proj0: [2*rank, hidden_size] -> extract gate/up at indices 0/1
    
    For proj1 (cola_b equivalent, Cola only):
    - qkv_proj1: [3, rank, out_features] (BatchedTensorParallelColumnLinear)
      * Contains q, k, v along batch dimension (first dim)
      * q: [0, :, :]
      * k: [1, :, :]
      * v: [2, :, :]
    - gate_up_proj1: [2, rank, out_features] -> extract gate/up at indices 0/1
    
    Note: Vanilla model has separate proj1 layers (q_proj1, k_proj1, etc.), so no extraction needed.
    """
    if is_proj0:
        # proj0: grouped along output dimension
        # PyTorch Linear stores as [out_features, in_features]
        if "qkv" in proj_name:
            rank = weight_tensor.shape[0] // 3
            if proj_idx == 0:  # q
                return weight_tensor[0:rank, :].t()  # Extract q slice and transpose to [hidden_size, rank]
            elif proj_idx == 1:  # k
                return weight_tensor[rank:2*rank, :].t()
            elif proj_idx == 2:  # v
                return weight_tensor[2*rank:3*rank, :].t()
        elif "gate_up" in proj_name or "gate" in proj_name or "up" in proj_name:
            rank = weight_tensor.shape[0] // 2
            if proj_idx == 0:  # gate
                return weight_tensor[0:rank, :].t()
            elif proj_idx == 1:  # up
                return weight_tensor[rank:2*rank, :].t()
    else:
        # proj1: grouped along batch dimension (first dim)
        if len(weight_tensor.shape) == 3:
            if "qkv" in proj_name:
                # Check which dimension is rank
                if weight_tensor.shape[1] < weight_tensor.shape[2]:
                    # [3, rank, out_features]
                    return weight_tensor[proj_idx, :, :].t()
                else:
                    # [3, out_features, rank]
                    return weight_tensor[proj_idx, :, :].t()
            elif "gate_up" in proj_name or "gate" in proj_name or "up" in proj_name:
                if weight_tensor.shape[1] < weight_tensor.shape[2]:
                    # [2, rank, out_features]
                    return weight_tensor[proj_idx, :, :].t()
                else:
                    # [2, out_features, rank]
                    return weight_tensor[proj_idx, :, :].t()
    
    return weight_tensor


def compare_params(basic_params: Dict, other_params: Dict, model_name: str, is_vanilla: bool = True, rtol=1e-3, atol=1e-5) -> Dict:
    """
    Compare parameters between Basic and other model.
    
    HOW GROUPING WORKS:
    ===================
    
    Basic model structure:
    - Separate layers: q_proj.cola_a, q_proj.cola_b, k_proj.cola_a, k_proj.cola_b, etc.
    - Each is a separate tensor: [in_features, rank] or [rank, out_features]
    
    Vanilla/Cola model structure:
    - Grouped GEMMs for efficiency:
      * qkv_proj0: [3*rank, hidden_size] - combines q, k, v cola_a projections
      * qkv_proj1: [3, rank, out_features] (Cola) or separate q_proj1, k_proj1, v_proj1 (Vanilla)
      * gate_up_proj0: [2*rank, hidden_size] - combines gate, up cola_a projections
      * gate_up_proj1: [2, rank, out_features] (Cola) or separate gate_proj1, up_proj1 (Vanilla)
    
    Extraction process:
    1. For proj0 (cola_a equivalent):
       - qkv_proj0: [3*rank, hidden_size] -> extract slices [0:rank, :], [rank:2*rank, :], [2*rank:3*rank, :]
       - gate_up_proj0: [2*rank, hidden_size] -> extract slices [0:rank, :], [rank:2*rank, :]
       - Then transpose to match Basic format [hidden_size, rank]
    
    2. For proj1 (cola_b equivalent):
       - Cola: qkv_proj1 is [3, rank, out_features] -> extract [0, :, :], [1, :, :], [2, :, :]
       - Vanilla: Separate layers, no extraction needed
    """
    results = {
        "missing_in_other": [],
        "shape_mismatches": [],
        "matches": [],
        "diffs": [],
        "weight_samples": [],  # Store samples for printing
    }
    
    # Process all basic weights
    for basic_name, basic_weight in basic_params.items():
        if ".cola_a" not in basic_name and ".cola_b" not in basic_name:
            # Not a CoLA weight, skip or handle separately
            continue
        
        # Map basic name to other name (without prefix)
        other_name_no_prefix = map_basic_to_other_weight_name(basic_name, is_vanilla=is_vanilla)
        
        if other_name_no_prefix is None:
            results["missing_in_other"].append(basic_name)
            continue
        
        # Try to find the actual key in other_params
        if other_name_no_prefix not in other_params:
            # Search for partial match
            matching_keys = [k for k in other_params.keys() if other_name_no_prefix in k or k.endswith(other_name_no_prefix)]
            if matching_keys:
                other_name = matching_keys[0]
            else:
                results["missing_in_other"].append(basic_name)
                continue
        else:
            other_name = other_name_no_prefix
        
        other_weight_raw = other_params[other_name]
        
        # Extract individual weight from grouped GEMM if needed
        is_cola_a = ".cola_a" in basic_name
        is_cola_b = ".cola_b" in basic_name
        
        # Determine which projection we're comparing and if it's grouped
        proj_name = None
        proj_idx = None
        needs_extraction = False
        
        if "q_proj" in basic_name:
            proj_idx = 0
            if is_cola_a:
                proj_name = "qkv"
                needs_extraction = True
            else:  # is_cola_b
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "qkv"
                    needs_extraction = True
        elif "k_proj" in basic_name:
            proj_idx = 1
            if is_cola_a:
                proj_name = "qkv"
                needs_extraction = True
            else:
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "qkv"
                    needs_extraction = True
        elif "v_proj" in basic_name:
            proj_idx = 2
            if is_cola_a:
                proj_name = "qkv"
                needs_extraction = True
            else:
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "qkv"
                    needs_extraction = True
        elif "gate_proj" in basic_name:
            proj_idx = 0
            if is_cola_a:
                proj_name = "gate_up"
                needs_extraction = True
            else:
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "gate_up"
                    needs_extraction = True
        elif "up_proj" in basic_name:
            proj_idx = 1
            if is_cola_a:
                proj_name = "gate_up"
                needs_extraction = True
            else:
                if is_vanilla:
                    needs_extraction = False
                else:
                    proj_name = "gate_up"
                    needs_extraction = True
        
        # Extract from grouped tensor if needed
        if needs_extraction and proj_name and ("qkv" in proj_name or "gate_up" in proj_name):
            # Debug: print shape before extraction
            if dist.get_rank() == 0 and "q_proj" in basic_name and is_cola_a:
                print(f"\nDEBUG: Extracting from grouped tensor")
                print(f"  Basic name: {basic_name}, Basic shape: {basic_weight.shape}")
                print(f"  Grouped tensor name: {other_name}, Grouped shape: {other_weight_raw.shape}")
                print(f"  Proj name: {proj_name}, Proj idx: {proj_idx}, Is proj0: {is_cola_a}")
            other_weight = extract_cola_weight_from_grouped(other_weight_raw, proj_name, proj_idx, is_cola_a)
            # Debug: print shape after extraction
            if dist.get_rank() == 0 and "q_proj" in basic_name and is_cola_a:
                print(f"  Extracted shape: {other_weight.shape}")
        else:
            other_weight = other_weight_raw
        
        # Handle transpose: PyTorch Linear stores weights as [out_features, in_features]
        # Basic cola_a is [in_features, rank] = [hidden_size, rank]
        # After extraction from grouped, we have [rank, hidden_size], so we transpose
        # But if shapes already match, no transpose needed
        if basic_weight.shape == tuple(reversed(other_weight.shape)):
            if dist.get_rank() == 0 and "q_proj" in basic_name and is_cola_a:
                print(f"  Transposing: {other_weight.shape} -> {other_weight.t().shape}")
            other_weight = other_weight.t()
        
        # Compare shapes
        if basic_weight.shape != other_weight.shape:
            basic_std = basic_weight.std().item()
            other_std = other_weight.std().item()
            results["shape_mismatches"].append({
                "basic_name": basic_name,
                "other_name": other_name,
                "basic_shape": basic_weight.shape,
                "other_shape": other_weight.shape,
                "basic_std": basic_std,
                "other_std": other_std,
            })
            continue
        
        # Calculate statistics
        diff = basic_weight - other_weight
        max_abs_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        relative_l2_diff = (diff.norm() / basic_weight.norm()).item() if basic_weight.norm() > 0 else float('inf')
        is_close = torch.allclose(basic_weight, other_weight, rtol=rtol, atol=atol)
        
        # Calculate std of each weight tensor
        basic_std = basic_weight.std().item()
        other_std = other_weight.std().item()
        
        # Store sample values for top differences
        sample_size = min(10, basic_weight.numel())
        basic_sample = basic_weight.flatten()[:sample_size].tolist()
        other_sample = other_weight.flatten()[:sample_size].tolist()
        
        result = {
            "basic_name": basic_name,
            "other_name": other_name,
            "max_abs_diff": max_abs_diff,
            "relative_l2_diff": relative_l2_diff,
            "basic_std": basic_std,
            "other_std": other_std,
            "std_diff": abs(basic_std - other_std),
            "basic_sample": basic_sample,
            "other_sample": other_sample,
            "needs_extraction": needs_extraction,
            "grouped_shape": other_weight_raw.shape if needs_extraction else None,
        }
        
        if is_close:
            results["matches"].append(result)
        else:
            results["diffs"].append(result)
    
    return results


def print_comparison(results: Dict, model_name: str):
    """Print comparison results"""
    print(f"\n{'='*80}")
    print(f"Comparison: Basic vs {model_name}")
    print(f"{'='*80}")
    
    print(f"\nMissing in {model_name}: {len(results['missing_in_other'])}")
    for name in results['missing_in_other'][:10]:  # Show first 10
        print(f"  - {name}")
    if len(results['missing_in_other']) > 10:
        print(f"  ... and {len(results['missing_in_other']) - 10} more")
    
    print(f"\nShape Mismatches: {len(results['shape_mismatches'])}")
    for mismatch in results['shape_mismatches'][:10]:
        print(f"  - {mismatch['basic_name']}")
        print(f"    Basic: {mismatch['basic_shape']}, {model_name}: {mismatch['other_shape']}")
    if len(results['shape_mismatches']) > 10:
        print(f"  ... and {len(results['shape_mismatches']) - 10} more")
    
    print(f"\nMatches: {len(results['matches'])}")
    print(f"Differences: {len(results['diffs'])}")
    
    if results['diffs']:
        print(f"\nTop 10 Differences (by max_abs_diff):")
        sorted_diffs = sorted(results['diffs'], key=lambda x: x['max_abs_diff'], reverse=True)
        print(f"{'Basic Name':<50} {'Max Diff':<15} {'Rel L2':<15} {'Basic Std':<15} {'Other Std':<15} {'Std Diff':<15}")
        print("-" * 120)
        for diff in sorted_diffs[:10]:
            print(f"{diff['basic_name']:<50} {diff['max_abs_diff']:<15.6e} {diff['relative_l2_diff']:<15.6e} "
                  f"{diff['basic_std']:<15.6e} {diff['other_std']:<15.6e} {diff['std_diff']:<15.6e}")
        
        # Print weight value samples for top 3 differences
        print(f"\n{'='*80}")
        print("WEIGHT VALUE SAMPLES (Top 3 differences)")
        print(f"{'='*80}")
        for i, diff in enumerate(sorted_diffs[:3]):
            print(f"\n{i+1}. {diff['basic_name']}")
            print(f"   Mapped to: {diff['other_name']}")
            if diff.get('needs_extraction'):
                print(f"   Extracted from grouped tensor shape: {diff['grouped_shape']}")
            print(f"   Basic std: {diff['basic_std']:.6e}, Other std: {diff['other_std']:.6e} ({diff['std_diff']/max(diff['basic_std'], 1e-10)*100:.2f}% diff)")
            print(f"   First 10 values:")
            print(f"     Basic:  {[f'{x:.6f}' for x in diff['basic_sample'][:10]]}")
            print(f"     Other:  {[f'{x:.6f}' for x in diff['other_sample'][:10]]}")
            print(f"     Diff:   {[f'{abs(a-b):.6f}' for a, b in zip(diff['basic_sample'][:10], diff['other_sample'][:10])]}")


def main():
    print("="*80)
    print("INITIALIZATION PARITY TEST")
    print("="*80)
    
    # Track if we initialized distributed
    initialized_dist = False
    
    try:
        # Set seed again to ensure reproducibility
        seed = 42
        set_random_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Create configs with same values
        print("\nCreating model configs...")
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
        
        # Initialize distributed environment
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            initialized_dist = True
        
        # Create full configs
        basic_full_config = create_test_config(basic_config, seed=seed)
        vanilla_full_config = create_test_config(vanilla_config, seed=seed)
        cola_full_config = create_test_config(cola_config, seed=seed)
        
        # Initialize models using DistributedTrainer (minimal setup, no training)
        print("\nInitializing models...")
        
        # Basic model
        set_random_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        basic_trainer = DistributedTrainer(
            config_or_config_file=basic_full_config,
            model_config_class=basic_config.__class__,
            model_class=BasicColaLlamaForTraining,
        )
        basic_model = basic_trainer.unwrapped_model
        
        # Vanilla model
        set_random_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        vanilla_trainer = DistributedTrainer(
            config_or_config_file=vanilla_full_config,
            model_config_class=vanilla_config.__class__,
            model_class=VanillaColaLlamaForTraining,
        )
        vanilla_model = vanilla_trainer.unwrapped_model
        
        # Cola model
        set_random_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cola_trainer = DistributedTrainer(
            config_or_config_file=cola_full_config,
            model_config_class=cola_config.__class__,
            model_class=ColaLlamaForTraining,
        )
        cola_model = cola_trainer.unwrapped_model
        
        print("Models initialized!")
        
        # Extract parameters
        print("\nExtracting parameters...")
        basic_params = get_param_dict(basic_model, "basic")
        vanilla_params = get_param_dict(vanilla_model, "vanilla")
        cola_params = get_param_dict(cola_model, "cola")
        
        print(f"Basic: {len(basic_params)} parameters")
        print(f"Vanilla: {len(vanilla_params)} parameters")
        print(f"Cola: {len(cola_params)} parameters")
        
        # Debug: Print some sample weight names
        if dist.get_rank() == 0:
            print("\nSample Basic weight names:")
            for name in list(basic_params.keys())[:5]:
                print(f"  {name}")
            print("\nSample Vanilla weight names:")
            for name in list(vanilla_params.keys())[:5]:
                print(f"  {name}")
            print("\nSample Cola weight names:")
            for name in list(cola_params.keys())[:5]:
                print(f"  {name}")
        
        # Compare
        print("\n" + "="*80)
        print("COMPARING PARAMETERS")
        print("="*80)
        
        vanilla_results = compare_params(basic_params, vanilla_params, "Vanilla", is_vanilla=True)
        cola_results = compare_params(basic_params, cola_params, "Cola", is_vanilla=False)
        
        print_comparison(vanilla_results, "Vanilla")
        print_comparison(cola_results, "Cola")
        
        # Determine PASS/FAIL
        # Note: Differences in values are expected due to RNG state differences
        # (Basic initializes in __init__, Vanilla/Cola initialize in init_model_randomly)
        # We check that std values are close (within 1%) to verify initialization scale is correct
        print("\n" + "="*80)
        print("TEST RESULT")
        print("="*80)
        
        # Check if std values match (within 1% tolerance)
        vanilla_std_pass = True
        cola_std_pass = True
        
        for diff in vanilla_results['diffs']:
            if diff['std_diff'] / max(diff['basic_std'], 1e-10) > 0.01:  # More than 1% difference
                vanilla_std_pass = False
                break
        
        for diff in cola_results['diffs']:
            if diff['std_diff'] / max(diff['basic_std'], 1e-10) > 0.01:  # More than 1% difference
                cola_std_pass = False
                break
        
        vanilla_pass = (
            len(vanilla_results['missing_in_other']) == 0 and
            len(vanilla_results['shape_mismatches']) == 0 and
            vanilla_std_pass
        )
        
        cola_pass = (
            len(cola_results['missing_in_other']) == 0 and
            len(cola_results['shape_mismatches']) == 0 and
            cola_std_pass
        )
        
        if vanilla_pass and cola_pass:
            print("✓ PASS: All parameters have matching shapes and std values!")
            print("  Note: Value differences are expected due to RNG state differences")
            print("  (Basic initializes in __init__, Vanilla/Cola initialize in init_model_randomly)")
            return 0
        else:
            print("✗ FAIL: Parameter mismatches found")
            if not vanilla_pass:
                print(f"  - Vanilla: {len(vanilla_results['missing_in_other'])} missing, "
                      f"{len(vanilla_results['shape_mismatches'])} shape mismatches")
                if not vanilla_std_pass:
                    print(f"    {len(vanilla_results['diffs'])} differences with std mismatch > 1%")
            if not cola_pass:
                print(f"  - Cola: {len(cola_results['missing_in_other'])} missing, "
                      f"{len(cola_results['shape_mismatches'])} shape mismatches")
                if not cola_std_pass:
                    print(f"    {len(cola_results['diffs'])} differences with std mismatch > 1%")
            return 1
    
    finally:
        # Clean up distributed process group
        if initialized_dist and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                # Ignore cleanup errors
                pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

