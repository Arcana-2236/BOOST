import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class LinearLayerFlopsCalculator:
    """Calculate FLOPs for linear layers in transformer decoder"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_gemm_flops(self, m: int, n: int, k: int) -> int:
        """
        Calculate FLOPs for GEMM operation: C = A @ B
        A: (m, k), B: (k, n), C: (m, n)
        FLOPs = 2 * m * n * k (multiply-add operations)
        """
        return 2 * m * n * k
    
    def calculate_attention_linear_flops(self, 
                                       seq_length: int,
                                       hidden_size: int,
                                       batch_size: int = 1,
                                       tp: int = 1) -> Dict[str, int]:
        """
        Calculate FLOPs for attention linear layer (d -> d)
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden dimension size (d)
            batch_size: Batch size
        
        Returns:
            FLOPs for attention linear layer
        """
        # Input tensor shape: (batch_size * seq_length, hidden_size)
        input_size = batch_size * seq_length
        
        # Linear layer: (batch_size * seq_length, hidden_size) @ (hidden_size, hidden_size)
        global_flops = self.calculate_gemm_flops(input_size, hidden_size, hidden_size)
        per_gpu_flops = global_flops // max(1, tp)
        return {"global": global_flops, "per_gpu": per_gpu_flops, "tp": tp}
    
    def calculate_mlp_linear_flops(self,
                                 seq_length: int,
                                 hidden_size: int,
                                 intermediate_size: int,
                                 batch_size: int = 1,
                                 tp: int = 1) -> Dict[str, int]:
        """
        Calculate FLOPs for MLP linear layer (d -> dff)
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden dimension size (d)
            intermediate_size: FFN intermediate size (dff)
            batch_size: Batch size
        
        Returns:
            FLOPs for MLP linear layer
        """
        # Input tensor shape: (batch_size * seq_length, hidden_size)
        input_size = batch_size * seq_length
        
        # Linear layer: (batch_size * seq_length, hidden_size) @ (hidden_size, intermediate_size)
        global_flops = self.calculate_gemm_flops(input_size, intermediate_size, hidden_size)
        per_gpu_flops = global_flops // max(1, tp)
        return {"global": global_flops, "per_gpu": per_gpu_flops, "tp": tp}


class LowRankLinearLayerFlopsCalculator:
    """Calculate FLOPs for low-rank linear layers in transformer decoder"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_gemm_flops(self, m: int, n: int, k: int) -> int:
        """
        Calculate FLOPs for GEMM operation: C = A @ B
        A: (m, k), B: (k, n), C: (m, n)
        FLOPs = 2 * m * n * k (multiply-add operations)
        """
        return 2 * m * n * k
    
    def calculate_low_rank_attention_linear_flops(self, 
                                                seq_length: int,
                                                hidden_size: int,
                                                rank: int,
                                                batch_size: int = 1,
                                                tp: int = 1) -> Dict[str, int]:
        """
        Calculate FLOPs for low-rank attention linear layer (d -> rank -> d)
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden dimension size (d)
            rank: Low-rank dimension
            batch_size: Batch size
        
        Returns:
            FLOPs for low-rank attention linear layer
        """
        # Input tensor shape: (batch_size * seq_length, hidden_size)
        input_size = batch_size * seq_length
        
        # First projection: (batch_size * seq_length, hidden_size) @ (hidden_size, rank)
        first_proj_flops = self.calculate_gemm_flops(input_size, rank, hidden_size)
        
        # Second projection: (batch_size * seq_length, rank) @ (rank, hidden_size)
        second_proj_flops = self.calculate_gemm_flops(input_size, hidden_size, rank)
        
        # Total low-rank FLOPs
        global_flops = first_proj_flops + second_proj_flops
        per_gpu_flops = global_flops // max(1, tp)
        return {"global": global_flops, "per_gpu": per_gpu_flops, "tp": tp}
    
    def calculate_low_rank_mlp_linear_flops(self,
                                          seq_length: int,
                                          hidden_size: int,
                                          intermediate_size: int,
                                          rank: int,
                                          batch_size: int = 1,
                                          tp: int = 1) -> Dict[str, int]:
        """
        Calculate FLOPs for low-rank MLP linear layer (d -> rank -> dff)
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden dimension size (d)
            intermediate_size: FFN intermediate size (dff)
            rank: Low-rank dimension
            batch_size: Batch size
        
        Returns:
            FLOPs for low-rank MLP linear layer
        """
        # Input tensor shape: (batch_size * seq_length, hidden_size)
        input_size = batch_size * seq_length
        
        # First projection: (batch_size * seq_length, hidden_size) @ (hidden_size, rank)
        first_proj_flops = self.calculate_gemm_flops(input_size, rank, hidden_size)
        
        # Second projection: (batch_size * seq_length, rank) @ (rank, intermediate_size)
        second_proj_flops = self.calculate_gemm_flops(input_size, intermediate_size, rank)
        
        # Total low-rank FLOPs
        global_flops = first_proj_flops + second_proj_flops
        per_gpu_flops = global_flops // max(1, tp)
        return {"global": global_flops, "per_gpu": per_gpu_flops, "tp": tp}



class BenchmarkLinearLayers:
    """Benchmark real execution time of full-rank and low-rank linear layers."""

    def __init__(self, device: str = 'cuda', dtype: torch.dtype = torch.float16):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def warmup_gpu(self, iterations: int = 50):
        """Comprehensive GPU warmup to ensure stable benchmarking"""
        if self.device.type != 'cuda':
            print("    CPU mode - skipping GPU warmup")
            return
            
        print(f"    🔥 GPU Warmup ({iterations} iterations)...")
        
        # Create dummy tensors for warmup
        dummy_a = torch.randn(1024, 1024, device=self.device, dtype=self.dtype)
        dummy_b = torch.randn(1024, 1024, device=self.device, dtype=self.dtype)
        
        # Warmup with various operations
        for i in range(iterations):
            if i % 10 == 0:
                print(f"      Warmup progress: {i}/{iterations}")
            # Matrix multiplication warmup
            _ = dummy_a @ dummy_b
            # Memory allocation warmup
            _ = torch.randn(512, 512, device=self.device, dtype=self.dtype)
        
        self._sync()
        print("    ✅ GPU warmup complete")

    def _time_op(self, fn, iters: int = 50, warmup: int = 20) -> float:
        # Extended warmup to ensure GPU is fully warmed up
        print(f"    Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            fn()
        self._sync()
        
        # Additional sync to ensure all operations are complete
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed measurements
        print(f"    Timing ({iters} iterations)...")
        start = time.time()
        for _ in range(iters):
            fn()
        self._sync()
        end = time.time()
        return (end - start) / iters

    def benchmark_fullrank_attention(self, seq_length: int, hidden_size: int, batch_size: int, tp: int = 1) -> float:
        # (B*S, d) @ (d, d) with TP sharding
        m = batch_size * seq_length
        d = hidden_size
        d_per_tp = d // tp
        
        A = torch.randn(m, d, device=self.device, dtype=self.dtype)
        # Simulate TP: each rank processes d/tp columns
        W = torch.randn(d, d_per_tp, device=self.device, dtype=self.dtype)
        
        def op():
            return A @ W
        return self._time_op(op)

    def benchmark_fullrank_mlp(self, seq_length: int, hidden_size: int, intermediate_size: int, batch_size: int, tp: int = 1) -> float:
        # (B*S, d) @ (d, dff) with TP sharding
        m = batch_size * seq_length
        d = hidden_size
        dff = intermediate_size
        dff_per_tp = dff // tp
        
        A = torch.randn(m, d, device=self.device, dtype=self.dtype)
        # Simulate TP: each rank processes dff/tp columns (up-projection)
        W = torch.randn(d, dff_per_tp, device=self.device, dtype=self.dtype)
        
        def op():
            return A @ W
        return self._time_op(op)

    def benchmark_lowrank_attention(self, seq_length: int, hidden_size: int, rank: int, batch_size: int, tp: int = 1, tp_mode: str = 'vanilla') -> float:
        """
        Benchmark low-rank attention with different TP modes:
        - vanilla: Standard TP sharding (d -> r/tp -> d)
        - btp: Bottleneck aware TP sharding (d/TP -> r -> d/TP)
        """
        m = batch_size * seq_length
        d = hidden_size
        r = rank
        
        A = torch.randn(m, d, device=self.device, dtype=self.dtype)
        
        if tp_mode == 'vanilla':
            # Vanilla TP: shard first projection only (d -> r/tp -> d)
            r_per_tp = r // tp
            W1 = torch.randn(d, r_per_tp, device=self.device, dtype=self.dtype)
            W2 = torch.randn(r_per_tp, d, device=self.device, dtype=self.dtype)
            
            def op():
                Z = A @ W1
                return Z @ W2
                
        elif tp_mode == 'btp':
            # Bottleneck aware TP: shard input/output dimensions (d/tp -> r -> d/tp)
            d_per_tp = d // tp
            W1 = torch.randn(d_per_tp, r, device=self.device, dtype=self.dtype)
            W2 = torch.randn(r, d_per_tp, device=self.device, dtype=self.dtype)
            
            def op():
                # Process only d/tp columns of input
                A_shard = A[:, :d_per_tp]
                Z = A_shard @ W1
                return Z @ W2
        else:
            raise ValueError(f"Unknown TP mode: {tp_mode}")
            
        return self._time_op(op)

    def benchmark_lowrank_mlp(self, seq_length: int, hidden_size: int, intermediate_size: int, rank: int, batch_size: int, tp: int = 1, tp_mode: str = 'vanilla') -> float:
        """
        Benchmark low-rank MLP with different TP modes:
        - vanilla: Standard TP sharding (d -> r/tp -> dff)
        - btp: Bottleneck aware TP sharding (d/tp -> r -> dff/tp)
        """
        m = batch_size * seq_length
        d = hidden_size
        dff = intermediate_size
        r = rank
        
        A = torch.randn(m, d, device=self.device, dtype=self.dtype)
        
        if tp_mode == 'vanilla':
            # Vanilla TP: shard first projection only (d -> r/tp -> dff)
            r_per_tp = r // tp
            W1 = torch.randn(d, r_per_tp, device=self.device, dtype=self.dtype)
            W2 = torch.randn(r_per_tp, dff, device=self.device, dtype=self.dtype)
            
            def op():
                Z = A @ W1
                return Z @ W2
                
        elif tp_mode == 'btp':
            # Bottleneck aware TP: shard input and output dimensions (d/tp -> r -> dff/tp)
            d_per_tp = d // tp
            dff_per_tp = dff // tp
            W1 = torch.randn(d_per_tp, r, device=self.device, dtype=self.dtype)
            W2 = torch.randn(r, dff_per_tp, device=self.device, dtype=self.dtype)
            
            def op():
                # Process only d/tp columns of input
                A_shard = A[:, :d_per_tp]
                Z = A_shard @ W1
                return Z @ W2
        else:
            raise ValueError(f"Unknown TP mode: {tp_mode}")
            
        return self._time_op(op)

# Example usage and testing
if __name__ == "__main__":
    fullrank_calculator = LinearLayerFlopsCalculator()
    lowrank_calculator = LowRankLinearLayerFlopsCalculator()
    
    # Example configurations for different model sizes
    configs = {
        "LLaMA-3B": {
            "seq_length": 4096,
            "hidden_size": 3072,
            "intermediate_size": 8192
        },
        "LLaMA-7B": {
            "seq_length": 4096,
            "hidden_size": 4096,
            "intermediate_size": 11008
        },
        "LLaMA-13B": {
            "seq_length": 4096,
            "hidden_size": 5120,
            "intermediate_size": 13824
        }
    }
    
    print("🚀 Linear Layer FLOPs Calculator for Transformer")
    print("=" * 60)
    
    # Calculate FLOPs for attention and MLP linear layers
    batch_size = 4  # Set batch size here
    tp_degrees = [4]  # Test different TP degrees
    
    for model_name, config in configs.items():
        # Calculate rank as 0.25 of hidden dimension
        rank = int(config['hidden_size'] * 0.25)
        
        print(f"\n📊 {model_name} Configuration:")
        print(f"  Sequence Length: {config['seq_length']}")
        print(f"  Hidden Size (d): {config['hidden_size']}")
        print(f"  Intermediate Size (dff): {config['intermediate_size']}")
        print(f"  Rank (0.25 * d): {rank}")
        print(f"  Batch Size: {batch_size}")
        
        # Test different TP degrees for FLOPs calculations
        for tp in tp_degrees:
            # Skip if TP degree is larger than dimensions
            if tp > config['hidden_size'] or tp > config['intermediate_size'] or tp > rank:
                continue
                
            print(f"\n🔍 TP={tp} FLOPs Analysis:")
        
        # Full-rank calculations
        print(f"  FULL-RANK Linear Layers:")
        
        # Calculate attention linear layer FLOPs (d -> d)
        attention_flops = fullrank_calculator.calculate_attention_linear_flops(
            seq_length=config['seq_length'],
            hidden_size=config['hidden_size'],
            batch_size=batch_size,
            tp=tp
        )
        print(f"    Attention Linear Layer (d -> d): global {attention_flops['global']:,} | per_gpu {attention_flops['per_gpu']:,}")
        
        # Calculate MLP linear layer FLOPs (d -> dff)
        mlp_flops = fullrank_calculator.calculate_mlp_linear_flops(
            seq_length=config['seq_length'],
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            batch_size=batch_size,
            tp=tp
        )
        print(f"    MLP Linear Layer (d -> dff): global {mlp_flops['global']:,} | per_gpu {mlp_flops['per_gpu']:,}")
        
        # Low-rank calculations
        print(f"  LOW-RANK Linear Layers (rank = {rank}):")
        
        # Calculate low-rank attention linear layer FLOPs (d -> rank -> d)
        lowrank_attention_flops = lowrank_calculator.calculate_low_rank_attention_linear_flops(
            seq_length=config['seq_length'],
            hidden_size=config['hidden_size'],
            rank=rank,
            batch_size=batch_size,
            tp=tp
        )
        print(f"    Attention Linear Layer (d -> {rank} -> d): global {lowrank_attention_flops['global']:,} | per_gpu {lowrank_attention_flops['per_gpu']:,}")
        
        # Calculate low-rank MLP linear layer FLOPs (d -> rank -> dff)
        lowrank_mlp_flops = lowrank_calculator.calculate_low_rank_mlp_linear_flops(
            seq_length=config['seq_length'],
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            rank=rank,
            batch_size=batch_size,
            tp=tp
        )
        print(f"    MLP Linear Layer (d -> {rank} -> dff): global {lowrank_mlp_flops['global']:,} | per_gpu {lowrank_mlp_flops['per_gpu']:,}")
        
        # Calculate compression ratios
        attention_compression = attention_flops['global'] / lowrank_attention_flops['global']
        mlp_compression = mlp_flops['global'] / lowrank_mlp_flops['global']
        
        print(f"  Compression Ratios:")
        print(f"    Attention: {attention_compression:.2f}x")
        print(f"    MLP: {mlp_compression:.2f}x")

    # Generate operations per GPU table
    print("\n📊 Operations per GPU Table")
    print("=" * 80)
    
    # Initialize table data
    table_data = {
        'Fullrank': {'attn': {}, 'mlp': {}},
        'Vanilla TP': {'attn': {}, 'mlp': {}},
        'BTP': {'attn': {}, 'mlp': {}}
    }
    
    # Calculate operations per GPU for each model and TP mode
    for model_name, config in configs.items():
        rank = int(config['hidden_size'] * 0.25)
        
        for tp in tp_degrees:
            # Skip if TP degree is larger than dimensions
            if tp > config['hidden_size'] or tp > config['intermediate_size'] or tp > rank:
                continue
            
            # Full-rank operations per GPU
            attn_full_flops = fullrank_calculator.calculate_attention_linear_flops(
                seq_length=config['seq_length'],
                hidden_size=config['hidden_size'],
                batch_size=batch_size,
                tp=tp
            )
            mlp_full_flops = fullrank_calculator.calculate_mlp_linear_flops(
                seq_length=config['seq_length'],
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                batch_size=batch_size,
                tp=tp
            )
            
            # Vanilla TP operations per GPU
            attn_vanilla_flops = lowrank_calculator.calculate_low_rank_attention_linear_flops(
                seq_length=config['seq_length'],
                hidden_size=config['hidden_size'],
                rank=rank,
                batch_size=batch_size,
                tp=tp
            )
            mlp_vanilla_flops = lowrank_calculator.calculate_low_rank_mlp_linear_flops(
                seq_length=config['seq_length'],
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                rank=rank,
                batch_size=batch_size,
                tp=tp
            )
            
            # BTP operations per GPU (same as vanilla TP for FLOPs, but different execution pattern)
            attn_btp_flops = attn_vanilla_flops  # Same FLOPs, different execution
            mlp_btp_flops = mlp_vanilla_flops    # Same FLOPs, different execution
            
            # Store in table
            table_data['Fullrank']['attn'][model_name] = attn_full_flops['per_gpu']
            table_data['Fullrank']['mlp'][model_name] = mlp_full_flops['per_gpu']
            table_data['Vanilla TP']['attn'][model_name] = attn_vanilla_flops['per_gpu']
            table_data['Vanilla TP']['mlp'][model_name] = mlp_vanilla_flops['per_gpu']
            table_data['BTP']['attn'][model_name] = attn_btp_flops['per_gpu']
            table_data['BTP']['mlp'][model_name] = mlp_btp_flops['per_gpu']
    
    # Print table header
    print(f"{'TFLOPs per GPU':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
    print("-" * 75)
    
    # Define the order of rows
    table_rows = [
        ('Fullrank Attn', 'Fullrank', 'attn'),
        ('Fullrank MLP', 'Fullrank', 'mlp'),
        ('Vanilla TP Attn', 'Vanilla TP', 'attn'),
        ('Vanilla TP MLP', 'Vanilla TP', 'mlp'),
        ('BTP Attn', 'BTP', 'attn'),
        ('BTP MLP', 'BTP', 'mlp')
    ]
    
    # Print table rows
    for row_name, mode, layer_type in table_rows:
        print(f"{row_name:<25} ", end="")
        for model_name in ['LLaMA-3B', 'LLaMA-7B', 'LLaMA-13B']:
            if model_name in table_data[mode][layer_type]:
                tflops = table_data[mode][layer_type][model_name] / 1e12  # Convert to TFLOPs
                print(f"{tflops:.3f}".ljust(15), end="")
            else:
                print("N/A".ljust(15), end="")
        print()

    # Optional: benchmark real execution time for all models with different TP degrees
    print("\n⏱️  Benchmarking real execution time with Tensor Parallelism...")
    bench = BenchmarkLinearLayers(device='cuda', dtype=torch.float16)
    
    # Comprehensive GPU warmup before benchmarking
    bench.warmup_gpu(iterations=100)
    
    batch_size = 4  # Same batch size as FLOPs calculations
    
    # Initialize benchmark table data
    benchmark_table_data = {
        'Fullrank': {'attn': {}, 'mlp': {}},
        'Vanilla TP': {'attn': {}, 'mlp': {}},
        'BTP': {'attn': {}, 'mlp': {}}
    }
    
    for model_name, test in configs.items():
        rank = int(test['hidden_size'] * 0.25)
        print(f"\n🧪 Benchmark: {model_name} (d={test['hidden_size']}, dff={test['intermediate_size']}, r={rank})")
        
        for tp in tp_degrees:
            # Skip if TP degree is larger than hidden dimension
            if tp > test['hidden_size'] or tp > test['intermediate_size'] or tp > rank:
                continue
                
            print(f"\n  📊 TP={tp}:")
            
            # Full-rank benchmarks
            t_attn_full = bench.benchmark_fullrank_attention(test['seq_length'], test['hidden_size'], batch_size, tp)
            t_mlp_full = bench.benchmark_fullrank_mlp(test['seq_length'], test['hidden_size'], test['intermediate_size'], batch_size, tp)
            
            # Low-rank benchmarks - both vanilla TP and BTP
            t_attn_lr_vanilla = bench.benchmark_lowrank_attention(test['seq_length'], test['hidden_size'], rank, batch_size, tp, 'vanilla')
            t_mlp_lr_vanilla = bench.benchmark_lowrank_mlp(test['seq_length'], test['hidden_size'], test['intermediate_size'], rank, batch_size, tp, 'vanilla')
            
            t_attn_lr_btp = bench.benchmark_lowrank_attention(test['seq_length'], test['hidden_size'], rank, batch_size, tp, 'btp')
            t_mlp_lr_btp = bench.benchmark_lowrank_mlp(test['seq_length'], test['hidden_size'], test['intermediate_size'], rank, batch_size, tp, 'btp')
            
            print(f"    Full-rank Attention: {t_attn_full*1e3:.3f} ms")
            print(f"    Full-rank MLP:       {t_mlp_full*1e3:.3f} ms")
            print(f"    Low-rank Attention (Vanilla TP):  {t_attn_lr_vanilla*1e3:.3f} ms")
            print(f"    Low-rank MLP (Vanilla TP):         {t_mlp_lr_vanilla*1e3:.3f} ms")
            print(f"    Low-rank Attention (BTP):         {t_attn_lr_btp*1e3:.3f} ms")
            print(f"    Low-rank MLP (BTP):               {t_mlp_lr_btp*1e3:.3f} ms")
            
            # Store benchmark results in table data
            benchmark_table_data['Fullrank']['attn'][model_name] = t_attn_full * 1e3  # Convert to ms
            benchmark_table_data['Fullrank']['mlp'][model_name] = t_mlp_full * 1e3
            benchmark_table_data['Vanilla TP']['attn'][model_name] = t_attn_lr_vanilla * 1e3
            benchmark_table_data['Vanilla TP']['mlp'][model_name] = t_mlp_lr_vanilla * 1e3
            benchmark_table_data['BTP']['attn'][model_name] = t_attn_lr_btp * 1e3
            benchmark_table_data['BTP']['mlp'][model_name] = t_mlp_lr_btp * 1e3
            
            # Here compute speed up of BTP vs Vanilla TP
            print(f"    Speed up of BTP vs Vanilla TP:")
            print(f"      Attention: {t_attn_lr_btp / t_attn_lr_vanilla:.2f}x")
            print(f"      MLP: {t_mlp_lr_btp / t_mlp_lr_vanilla:.2f}x")

            # Here compute speed up of BTP vs Full-rank
            print(f"    Speed up of BTP vs Full-rank:")
            print(f"      Attention: {t_attn_lr_btp / t_attn_full:.2f}x")
            print(f"      MLP: {t_mlp_lr_btp / t_mlp_full:.2f}x")

    # Generate benchmark execution time table
    print("\n⏱️  Benchmark Execution Time Table (ms)")
    print("=" * 80)
    
    # Print table header
    print(f"{'Execution Time (ms)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
    print("-" * 75)
    
    # Define the order of rows
    table_rows = [
        ('Fullrank Attn', 'Fullrank', 'attn'),
        ('Fullrank MLP', 'Fullrank', 'mlp'),
        ('Vanilla TP Attn', 'Vanilla TP', 'attn'),
        ('Vanilla TP MLP', 'Vanilla TP', 'mlp'),
        ('BTP Attn', 'BTP', 'attn'),
        ('BTP MLP', 'BTP', 'mlp')
    ]
    
    # Print table rows
    for row_name, mode, layer_type in table_rows:
        print(f"{row_name:<25} ", end="")
        for model_name in ['LLaMA-3B', 'LLaMA-7B', 'LLaMA-13B']:
            if model_name in benchmark_table_data[mode][layer_type]:
                time_ms = benchmark_table_data[mode][layer_type][model_name]
                print(f"{time_ms:.3f}".ljust(15), end="")
            else:
                print("N/A".ljust(15), end="")
        print()

    # Generate LLaMA 7B specific hardware utilization table
    print("\n⚡ LLaMA 7B Hardware Utilization Table (%)")
    print("=" * 80)
    print("Utilization = (TFLOPs per GPU / Time in seconds) / A100 Peak (312 TFLOPs) * 100")
    print("TP=4, Different Batch Sizes")
    print("-" * 80)
    
    # LLaMA 7B configuration
    llama7b_config = configs["LLaMA-7B"]
    llama7b_rank = int(llama7b_config['hidden_size'] * 0.25)
    batch_sizes = [1, 2, 4, 8]
    tp_fixed = 4
    
    # Initialize table data for LLaMA 7B
    llama7b_table_data = {
        'Fullrank': {'attn': {}, 'mlp': {}},
        'Vanilla TP': {'attn': {}, 'mlp': {}},
        'BTP': {'attn': {}, 'mlp': {}}
    }
    
    # Calculate for different batch sizes
    for batch_size in batch_sizes:
        # Full-rank calculations
        attn_full_flops = fullrank_calculator.calculate_attention_linear_flops(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            batch_size=batch_size,
            tp=tp_fixed
        )
        mlp_full_flops = fullrank_calculator.calculate_mlp_linear_flops(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        # Vanilla TP calculations
        attn_vanilla_flops = lowrank_calculator.calculate_low_rank_attention_linear_flops(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            rank=llama7b_rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        mlp_vanilla_flops = lowrank_calculator.calculate_low_rank_mlp_linear_flops(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            rank=llama7b_rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        # BTP calculations (same FLOPs as vanilla TP)
        attn_btp_flops = attn_vanilla_flops
        mlp_btp_flops = mlp_vanilla_flops
        
        # Store in table
        llama7b_table_data['Fullrank']['attn'][batch_size] = attn_full_flops['per_gpu']
        llama7b_table_data['Fullrank']['mlp'][batch_size] = mlp_full_flops['per_gpu']
        llama7b_table_data['Vanilla TP']['attn'][batch_size] = attn_vanilla_flops['per_gpu']
        llama7b_table_data['Vanilla TP']['mlp'][batch_size] = mlp_vanilla_flops['per_gpu']
        llama7b_table_data['BTP']['attn'][batch_size] = attn_btp_flops['per_gpu']
        llama7b_table_data['BTP']['mlp'][batch_size] = mlp_btp_flops['per_gpu']
    
    # Benchmark for different batch sizes
    print(f"\n🧪 Benchmarking LLaMA 7B with TP={tp_fixed}...")
    bench = BenchmarkLinearLayers(device='cuda', dtype=torch.float16)
    bench.warmup_gpu(iterations=100)
    
    llama7b_benchmark_data = {
        'Fullrank': {'attn': {}, 'mlp': {}},
        'Vanilla TP': {'attn': {}, 'mlp': {}},
        'BTP': {'attn': {}, 'mlp': {}}
    }
    
    for batch_size in batch_sizes:
        print(f"\n  📊 Batch Size={batch_size}:")
        
        # Full-rank benchmarks
        t_attn_full = bench.benchmark_fullrank_attention(llama7b_config['seq_length'], llama7b_config['hidden_size'], batch_size, tp_fixed)
        t_mlp_full = bench.benchmark_fullrank_mlp(llama7b_config['seq_length'], llama7b_config['hidden_size'], llama7b_config['intermediate_size'], batch_size, tp_fixed)
        
        # Low-rank benchmarks
        t_attn_lr_vanilla = bench.benchmark_lowrank_attention(llama7b_config['seq_length'], llama7b_config['hidden_size'], llama7b_rank, batch_size, tp_fixed, 'vanilla')
        t_mlp_lr_vanilla = bench.benchmark_lowrank_mlp(llama7b_config['seq_length'], llama7b_config['hidden_size'], llama7b_config['intermediate_size'], llama7b_rank, batch_size, tp_fixed, 'vanilla')
        
        t_attn_lr_btp = bench.benchmark_lowrank_attention(llama7b_config['seq_length'], llama7b_config['hidden_size'], llama7b_rank, batch_size, tp_fixed, 'btp')
        t_mlp_lr_btp = bench.benchmark_lowrank_mlp(llama7b_config['seq_length'], llama7b_config['hidden_size'], llama7b_config['intermediate_size'], llama7b_rank, batch_size, tp_fixed, 'btp')
        
        # Store benchmark results
        llama7b_benchmark_data['Fullrank']['attn'][batch_size] = t_attn_full * 1e3  # Convert to ms
        llama7b_benchmark_data['Fullrank']['mlp'][batch_size] = t_mlp_full * 1e3
        llama7b_benchmark_data['Vanilla TP']['attn'][batch_size] = t_attn_lr_vanilla * 1e3
        llama7b_benchmark_data['Vanilla TP']['mlp'][batch_size] = t_mlp_lr_vanilla * 1e3
        llama7b_benchmark_data['BTP']['attn'][batch_size] = t_attn_lr_btp * 1e3
        llama7b_benchmark_data['BTP']['mlp'][batch_size] = t_mlp_lr_btp * 1e3
        
        print(f"    Full-rank Attention: {t_attn_full*1e3:.3f} ms")
        print(f"    Full-rank MLP:       {t_mlp_full*1e3:.3f} ms")
        print(f"    Vanilla TP Attention: {t_attn_lr_vanilla*1e3:.3f} ms")
        print(f"    Vanilla TP MLP:       {t_mlp_lr_vanilla*1e3:.3f} ms")
        print(f"    BTP Attention:        {t_attn_lr_btp*1e3:.3f} ms")
        print(f"    BTP MLP:              {t_mlp_lr_btp*1e3:.3f} ms")
    
    # Print LLaMA 7B utilization table
    print(f"\n{'Hardware Utilization (%)':<25} {'1':<15} {'2':<15} {'4':<15} {'8':<15}")
    print("-" * 95)
    
    # Define the order of rows
    table_rows = [
        ('Fullrank Attn', 'Fullrank', 'attn'),
        ('Fullrank MLP', 'Fullrank', 'mlp'),
        ('Vanilla TP Attn', 'Vanilla TP', 'attn'),
        ('Vanilla TP MLP', 'Vanilla TP', 'mlp'),
        ('BTP Attn', 'BTP', 'attn'),
        ('BTP MLP', 'BTP', 'mlp')
    ]
    
    for row_name, mode, layer_type in table_rows:
        print(f"{row_name:<25} ", end="")
        for batch_size in [1, 2, 4, 8]:
            if batch_size in llama7b_table_data[mode][layer_type] and batch_size in llama7b_benchmark_data[mode][layer_type]:
                # Get TFLOPs per GPU
                tflops = llama7b_table_data[mode][layer_type][batch_size] / 1e12
                # Get execution time in seconds
                time_s = llama7b_benchmark_data[mode][layer_type][batch_size] / 1000
                # Calculate utilization
                utilization = (tflops / time_s) / 312 * 100
                print(f"{utilization:.1f}".ljust(15), end="")
            else:
                print("N/A".ljust(15), end="")
        print()

    # Generate hardware FLOPs utilization table
    print("\n⚡ Hardware FLOPs Utilization Table (%)")
    print("=" * 80)
    print("Utilization = (TFLOPs per GPU / Time in seconds) / A100 Peak (312 TFLOPs) * 100")
    print("-" * 80)
    
    # Print table header
    print(f"{'Hardware Utilization (%)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
    print("-" * 75)
    
    # Define the order of rows
    table_rows = [
        ('Fullrank Attn', 'Fullrank', 'attn'),
        ('Fullrank MLP', 'Fullrank', 'mlp'),
        ('Vanilla TP Attn', 'Vanilla TP', 'attn'),
        ('Vanilla TP MLP', 'Vanilla TP', 'mlp'),
        ('BTP Attn', 'BTP', 'attn'),
        ('BTP MLP', 'BTP', 'mlp')
    ]
    
    # Print table rows
    for row_name, mode, layer_type in table_rows:
        print(f"{row_name:<25} ", end="")
        for model_name in ['LLaMA-3B', 'LLaMA-7B', 'LLaMA-13B']:
            if model_name in table_data[mode][layer_type] and model_name in benchmark_table_data[mode][layer_type]:
                # Get TFLOPs per GPU
                tflops = table_data[mode][layer_type][model_name] / 1e12
                # Get execution time in seconds
                time_s = benchmark_table_data[mode][layer_type][model_name] / 1000  # Convert ms to seconds
                # Calculate utilization: TFLOPs/s / 312 TFLOPs * 100
                utilization = (tflops / time_s) / 312 * 100
                print(f"{utilization:.1f}".ljust(15), end="")
            else:
                print("N/A".ljust(15), end="")
        print()

    # Save summary tables to a text file for report/README usage.
    output_path = Path(__file__).with_name("comp_efficiency_tables.txt")
    lines = []

    lines.append("Operations per GPU Table")
    lines.append("=" * 80)
    lines.append(f"{'TFLOPs per GPU':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
    lines.append("-" * 75)
    for row_name, mode, layer_type in table_rows:
        row = f"{row_name:<25} "
        for model_name in ['LLaMA-3B', 'LLaMA-7B', 'LLaMA-13B']:
            if model_name in table_data[mode][layer_type]:
                tflops = table_data[mode][layer_type][model_name] / 1e12
                row += f"{tflops:.3f}".ljust(15)
            else:
                row += "N/A".ljust(15)
        lines.append(row)

    lines.append("")
    lines.append("⏱️  Benchmark Execution Time Table (ms)")
    lines.append("=" * 80)
    lines.append(f"{'Execution Time (ms)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
    lines.append("-" * 75)
    for row_name, mode, layer_type in table_rows:
        row = f"{row_name:<25} "
        for model_name in ['LLaMA-3B', 'LLaMA-7B', 'LLaMA-13B']:
            if model_name in benchmark_table_data[mode][layer_type]:
                row += f"{benchmark_table_data[mode][layer_type][model_name]:.3f}".ljust(15)
            else:
                row += "N/A".ljust(15)
        lines.append(row)

    lines.append("")
    lines.append(f"{'Hardware Utilization (%)':<25} {'1':<15} {'2':<15} {'4':<15} {'8':<15}")
    lines.append("-" * 95)
    for row_name, mode, layer_type in table_rows:
        row = f"{row_name:<25} "
        for batch_size in [1, 2, 4, 8]:
            if batch_size in llama7b_table_data[mode][layer_type] and batch_size in llama7b_benchmark_data[mode][layer_type]:
                tflops = llama7b_table_data[mode][layer_type][batch_size] / 1e12
                time_s = llama7b_benchmark_data[mode][layer_type][batch_size] / 1000
                utilization = (tflops / time_s) / 312 * 100
                row += f"{utilization:.1f}".ljust(15)
            else:
                row += "N/A".ljust(15)
        lines.append(row)

    lines.append("")
    lines.append("⚡ Hardware FLOPs Utilization Table (%)")
    lines.append("=" * 80)
    lines.append("Utilization = (TFLOPs per GPU / Time in seconds) / A100 Peak (312 TFLOPs) * 100")
    lines.append("-" * 80)
    lines.append(f"{'Hardware Utilization (%)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
    lines.append("-" * 75)
    for row_name, mode, layer_type in table_rows:
        row = f"{row_name:<25} "
        for model_name in ['LLaMA-3B', 'LLaMA-7B', 'LLaMA-13B']:
            if model_name in table_data[mode][layer_type] and model_name in benchmark_table_data[mode][layer_type]:
                tflops = table_data[mode][layer_type][model_name] / 1e12
                time_s = benchmark_table_data[mode][layer_type][model_name] / 1000
                utilization = (tflops / time_s) / 312 * 100
                row += f"{utilization:.1f}".ljust(15)
            else:
                row += "N/A".ljust(15)
        lines.append(row)

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\n📝 Saved summary tables to: {output_path}")
