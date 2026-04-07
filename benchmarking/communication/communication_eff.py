import torch
import torch.distributed as dist
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def is_rank_0() -> bool:
    """Check if current process is rank 0 or not in distributed mode"""
    return not dist.is_initialized() or dist.get_rank() == 0

class CommunicationVolumeCalculator:
    """Calculate communication volume for different tensor parallelism methods"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_bytes(self, tensor_shape: Tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> int:
        """Calculate bytes for a tensor"""
        element_size = 2 if dtype == torch.bfloat16 else 4  # bfloat16 = 2 bytes, float32 = 4 bytes
        return torch.tensor(tensor_shape).prod().item() * element_size
    
    def calculate_fullrank_communication_volume(self, 
                                             seq_length: int,
                                             hidden_size: int,
                                             intermediate_size: int,
                                             batch_size: int = 1,
                                             tp: int = 1,
                                             dtype: torch.dtype = torch.bfloat16) -> Dict[str, int]:
        """
        Calculate communication volume for full-rank tensor parallelism
        
        Args:
            seq_length: Sequence length
            hidden_size: Hidden dimension size (d)
            intermediate_size: FFN intermediate size (dff)
            batch_size: Batch size
            tp: Tensor parallelism degree
            dtype: Data type
            
        Returns:
            Dictionary with 'attn', 'mlp', and 'total' communication volumes in bytes
        """
        # Attention: (batch_size, seq_length, hidden_size) -> all-reduce
        attn_shape = (batch_size, seq_length, hidden_size)
        attn_bytes = self.calculate_bytes(attn_shape, dtype)
        
        # MLP: (batch_size, seq_length, hidden_size) -> all-reduce
        mlp_shape = (batch_size, seq_length, hidden_size)
        mlp_bytes = self.calculate_bytes(mlp_shape, dtype)

        total_bytes = attn_bytes + mlp_bytes
        return {"attn": attn_bytes, "mlp": mlp_bytes, "total": total_bytes}
    
    def calculate_vanilla_tp_communication_volume(self,
                                                 seq_length: int,
                                                 hidden_size: int,
                                                 intermediate_size: int,
                                                 rank: int,
                                                 batch_size: int = 1,
                                                 tp: int = 1,
                                                 dtype: torch.dtype = torch.bfloat16) -> Dict[str, int]:
        """
        Calculate communication volume for vanilla tensor parallelism
        
        Vanilla TP: (d -> r/tp -> d) and (d -> r/tp -> dff)
        
        Returns:
            Dictionary with 'attn', 'mlp', and 'total' communication volumes in bytes
        """
        # Attention
        qkv_attn_shape = (batch_size, seq_length, hidden_size)
        qkv_attn_bytes = 3 * self.calculate_bytes(qkv_attn_shape, dtype)
        o_attn_shape = (batch_size, seq_length, hidden_size)
        o_attn_bytes = self.calculate_bytes(o_attn_shape, dtype)
        attn_total_bytes = qkv_attn_bytes + o_attn_bytes

        # MLP
        gate_up_shape = (batch_size, seq_length, intermediate_size)
        gate_up_bytes = 2 * self.calculate_bytes(gate_up_shape, dtype)
        down_shape = (batch_size, seq_length, hidden_size)
        down_bytes = self.calculate_bytes(down_shape, dtype)
        mlp_total_bytes = gate_up_bytes + down_bytes

        total_bytes = attn_total_bytes + mlp_total_bytes
        return {"attn": attn_total_bytes, "mlp": mlp_total_bytes, "total": total_bytes}
    
    
    def calculate_btp_communication_volume(self,
                                        seq_length: int,
                                        hidden_size: int,
                                        intermediate_size: int,
                                        rank: int,
                                        batch_size: int = 1,
                                        tp: int = 1,
                                        dtype: torch.dtype = torch.bfloat16) -> Dict[str, int]:
        """
        Calculate communication volume for bottleneck-aware tensor parallelism
        
        BTP: (d/tp -> r -> d/tp) and (d/tp -> r -> dff/tp)
        Communication happens after second projection
        
        Returns:
            Dictionary with 'attn', 'mlp', and 'total' communication volumes in bytes
        """
        # Attention communication (after second projection)
        attn_shape = (batch_size, seq_length, rank)
        attn_bytes = 4 * self.calculate_bytes(attn_shape, dtype)  # Q, K, V, O projections
        
        # MLP communication (after second projection)  
        mlp_shape = (batch_size, seq_length, rank)
        mlp_bytes = 3 * self.calculate_bytes(mlp_shape, dtype)  # Gate, Up, Down projections
        
        total_bytes = attn_bytes + mlp_bytes
        return {"attn": attn_bytes, "mlp": mlp_bytes, "total": total_bytes}


class CommunicationBenchmark:
    """Benchmark real communication time for different tensor parallelism methods"""
    
    def __init__(self, device: str = 'cuda', dtype: torch.dtype = torch.bfloat16):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Initialize distributed if not already done
        if not dist.is_initialized():
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend='nccl')
            except:
                print("Warning: Distributed not initialized, using single GPU simulation")
    
    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
    
    def warmup_communication(self, iterations: int = 10):
        """Warmup communication operations"""
        if not dist.is_initialized():
            print("    Single GPU mode - skipping communication warmup")
            return
            
        if dist.get_rank() == 0:
            print(f"    🔥 Communication Warmup ({iterations} iterations)...")
        
        # Create dummy tensors for warmup
        dummy_tensor = torch.randn(1024, 1024, device=self.device, dtype=self.dtype)
        
        for i in range(iterations):
            if i % 5 == 0 and dist.get_rank() == 0:
                print(f"      Warmup progress: {i}/{iterations}")
            dist.all_reduce(dummy_tensor, op=dist.ReduceOp.SUM)
        
        self._sync()
        if dist.get_rank() == 0:
            print("    ✅ Communication warmup complete")
    
    def _time_communication(self, fn, iters: int = 20, warmup: int = 5) -> float:
        """Time communication operations"""
        # Warmup
        for _ in range(warmup):
            fn()
        self._sync()
        
        # Timed measurements
        start = time.time()
        for _ in range(iters):
            fn()
        self._sync()
        end = time.time()
        return (end - start) / iters
    
    def benchmark_fullrank_communication(self, 
                                      seq_length: int,
                                      hidden_size: int,
                                      intermediate_size: int,
                                      batch_size: int,
                                      tp: int = 1) -> Dict[str, float]:
        """Benchmark full-rank communication"""
        # Create tensors
        attn_tensor = torch.randn(batch_size, seq_length, hidden_size, 
                                device=self.device, dtype=self.dtype)
        mlp_tensor = torch.randn(batch_size, seq_length, hidden_size,
                               device=self.device, dtype=self.dtype)
        
        def attn_comm_op():
            dist.all_reduce(attn_tensor, op=dist.ReduceOp.SUM)
        
        def mlp_comm_op():
            dist.all_reduce(mlp_tensor, op=dist.ReduceOp.SUM)
        
        attn_time = self._time_communication(attn_comm_op)
        mlp_time = self._time_communication(mlp_comm_op)
        
        return {"attn": attn_time, "mlp": mlp_time, "total": attn_time + mlp_time}
    
    def benchmark_vanilla_tp_communication(self,
                                        seq_length: int,
                                        hidden_size: int,
                                        intermediate_size: int,
                                        rank: int,
                                        batch_size: int,
                                        tp: int = 1) -> Dict[str, float]:
        """Benchmark vanilla TP communication"""
        # Attention tensors
        qkv_tensor = torch.randn(3 * batch_size, seq_length, hidden_size,
                                device=self.device, dtype=self.dtype)
        o_tensor = torch.randn(batch_size, seq_length, hidden_size,
                               device=self.device, dtype=self.dtype)
        
        # MLP tensors
        gate_up_tensor = torch.randn(2 * batch_size, seq_length, intermediate_size,
                                device=self.device, dtype=self.dtype)
        down_tensor = torch.randn(batch_size, seq_length, hidden_size,
                               device=self.device, dtype=self.dtype)
        
        def attn_comm_op():
            dist.all_reduce(qkv_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(o_tensor, op=dist.ReduceOp.SUM)
        
        def mlp_comm_op():
            dist.all_reduce(gate_up_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(down_tensor, op=dist.ReduceOp.SUM)
        
        attn_time = self._time_communication(attn_comm_op)
        mlp_time = self._time_communication(mlp_comm_op)
        
        return {"attn": attn_time, "mlp": mlp_time, "total": attn_time + mlp_time}
    
    def benchmark_btp_communication(self,
                                seq_length: int,
                                hidden_size: int,
                                intermediate_size: int,
                                rank: int,
                                batch_size: int,
                                tp: int = 1) -> Dict[str, float]:
        """Benchmark BTP communication"""
        # Create tensors for second projection outputs
        qkv_tensor = torch.randn(3 * batch_size, seq_length, rank,
                                device=self.device, dtype=self.dtype)
        o_tensor = torch.randn(batch_size, seq_length, rank,
                               device=self.device, dtype=self.dtype)
        gate_up_tensor = torch.randn(2 * batch_size, seq_length, rank,
                                 device=self.device, dtype=self.dtype)
        down_tensor = torch.randn(batch_size, seq_length, rank,
                               device=self.device, dtype=self.dtype)
        
        def attn_comm_op():
            dist.all_reduce(qkv_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(o_tensor, op=dist.ReduceOp.SUM)
        
        def mlp_comm_op():
            dist.all_reduce(gate_up_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(down_tensor, op=dist.ReduceOp.SUM)
        
        attn_time = self._time_communication(attn_comm_op)
        mlp_time = self._time_communication(mlp_comm_op)
        
        return {"attn": attn_time, "mlp": mlp_time, "total": attn_time + mlp_time}


# Example usage and testing
if __name__ == "__main__":
    comm_calculator = CommunicationVolumeCalculator()
    comm_benchmark = CommunicationBenchmark(device='cuda', dtype=torch.bfloat16)
    
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
    
    if is_rank_0():
        print("📡 Communication Volume Calculator for Tensor Parallelism")
        print("=" * 70)
    
    # Calculate communication volume for different models and TP degrees
    batch_size = 4
    tp_degrees = [4]
    
    # Initialize table data
    comm_volume_data = {
        'Fullrank': {'attn': {}, 'mlp': {}},
        'Vanilla TP': {'attn': {}, 'mlp': {}},
        'BTP': {'attn': {}, 'mlp': {}}
    }
    
    comm_time_data = {
        'Fullrank': {'attn': {}, 'mlp': {}},
        'Vanilla TP': {'attn': {}, 'mlp': {}},
        'BTP': {'attn': {}, 'mlp': {}}
    }
    
    for model_name, config in configs.items():
        rank = int(config['hidden_size'] * 0.25)
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\n📊 {model_name} Communication Analysis:")
            print(f"  Sequence Length: {config['seq_length']}")
            print(f"  Hidden Size (d): {config['hidden_size']}")
            print(f"  Intermediate Size (dff): {config['intermediate_size']}")
            print(f"  Rank (0.25 * d): {rank}")
            print(f"  Batch Size: {batch_size}")
        
        for tp in tp_degrees:
            # Skip if TP degree is larger than dimensions
            if tp > config['hidden_size'] or tp > config['intermediate_size'] or tp > rank:
                continue
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\n🔍 TP={tp} Communication Analysis:")
            
            # Full-rank communication volume
            fullrank_comm = comm_calculator.calculate_fullrank_communication_volume(
                seq_length=config['seq_length'],
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                batch_size=batch_size,
                tp=tp
            )
            
            # Vanilla TP communication volume
            vanilla_tp_comm = comm_calculator.calculate_vanilla_tp_communication_volume(
                seq_length=config['seq_length'],
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                rank=rank,
                batch_size=batch_size,
                tp=tp
            )
            
            # BTP communication volume
            btp_comm = comm_calculator.calculate_btp_communication_volume(
                seq_length=config['seq_length'],
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                rank=rank,
                batch_size=batch_size,
                tp=tp
            )
            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"  Fullrank Communication:")
                print(f"    Attention: {fullrank_comm['attn']:,} bytes ({fullrank_comm['attn']/1024/1024:.2f} MB)")
                print(f"    MLP: {fullrank_comm['mlp']:,} bytes ({fullrank_comm['mlp']/1024/1024:.2f} MB)")
                print(f"    Total: {fullrank_comm['total']:,} bytes ({fullrank_comm['total']/1024/1024:.2f} MB)")
                
                print(f"  Vanilla TP Communication:")
                print(f"    Attention: {vanilla_tp_comm['attn']:,} bytes ({vanilla_tp_comm['attn']/1024/1024:.2f} MB)")
                print(f"    MLP: {vanilla_tp_comm['mlp']:,} bytes ({vanilla_tp_comm['mlp']/1024/1024:.2f} MB)")
                print(f"    Total: {vanilla_tp_comm['total']:,} bytes ({vanilla_tp_comm['total']/1024/1024:.2f} MB)")
                
                print(f"  BTP Communication:")
                print(f"    Attention: {btp_comm['attn']:,} bytes ({btp_comm['attn']/1024/1024:.2f} MB)")
                print(f"    MLP: {btp_comm['mlp']:,} bytes ({btp_comm['mlp']/1024/1024:.2f} MB)")
                print(f"    Total: {btp_comm['total']:,} bytes ({btp_comm['total']/1024/1024:.2f} MB)")
            
            # Store in table data
            comm_volume_data['Fullrank']['attn'][model_name] = fullrank_comm['attn']
            comm_volume_data['Fullrank']['mlp'][model_name] = fullrank_comm['mlp']
            comm_volume_data['Vanilla TP']['attn'][model_name] = vanilla_tp_comm['attn']
            comm_volume_data['Vanilla TP']['mlp'][model_name] = vanilla_tp_comm['mlp']
            comm_volume_data['BTP']['attn'][model_name] = btp_comm['attn']
            comm_volume_data['BTP']['mlp'][model_name] = btp_comm['mlp']
    
    table_rows = [
        ('Fullrank', 'Fullrank'),
        ('Vanilla TP', 'Vanilla TP'),
        ('BTP', 'BTP')
    ]
    model_order = ['LLaMA-3B', 'LLaMA-7B', 'LLaMA-13B']

    # Generate communication volume table
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n📊 Communication Volume Table (MB)")
        print("=" * 80)
    
        # Print table header
        print(f"{'Communication Volume (MB)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
        print("-" * 75)
    
        # Print table rows
        for row_name, mode in table_rows:
            print(f"{row_name:<25} ", end="")
            for model_name in model_order:
                if model_name in comm_volume_data[mode]['attn']:
                    total_mb = (comm_volume_data[mode]['attn'][model_name] + comm_volume_data[mode]['mlp'][model_name]) / (1024 * 1024)
                    print(f"{total_mb:.2f}".ljust(15), end="")
                else:
                    print("N/A".ljust(15), end="")
            print()
    
    # Communication benchmarking
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n⏱️  Communication Benchmarking...")
    comm_benchmark.warmup_communication(iterations=20)
    
    batch_size = 4
    tp_fixed = 4  # Fixed TP for benchmarking
    
    for model_name, config in configs.items():
        rank = int(config['hidden_size'] * 0.25)     
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\n🧪 Communication Benchmark: {model_name} (TP={tp_fixed})")
        
        # Benchmark different communication methods
        t_fullrank = comm_benchmark.benchmark_fullrank_communication(
            seq_length=config['seq_length'],
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        t_vanilla_tp = comm_benchmark.benchmark_vanilla_tp_communication(
            seq_length=config['seq_length'],
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            rank=rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        t_btp = comm_benchmark.benchmark_btp_communication(
            seq_length=config['seq_length'],
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            rank=rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"  Fullrank Communication:")
            print(f"    Attention: {t_fullrank['attn']*1e3:.3f} ms")
            print(f"    MLP: {t_fullrank['mlp']*1e3:.3f} ms")
            print(f"    Total: {t_fullrank['total']*1e3:.3f} ms")
            
            print(f"  Vanilla TP Communication:")
            print(f"    Attention: {t_vanilla_tp['attn']*1e3:.3f} ms")
            print(f"    MLP: {t_vanilla_tp['mlp']*1e3:.3f} ms")
            print(f"    Total: {t_vanilla_tp['total']*1e3:.3f} ms")
            
            print(f"  BTP Communication:")
            print(f"    Attention: {t_btp['attn']*1e3:.3f} ms")
            print(f"    MLP: {t_btp['mlp']*1e3:.3f} ms")
            print(f"    Total: {t_btp['total']*1e3:.3f} ms")
        
        # Store benchmark results
        comm_time_data['Fullrank']['attn'][model_name] = t_fullrank['attn'] * 1e3
        comm_time_data['Fullrank']['mlp'][model_name] = t_fullrank['mlp'] * 1e3
        comm_time_data['Vanilla TP']['attn'][model_name] = t_vanilla_tp['attn'] * 1e3
        comm_time_data['Vanilla TP']['mlp'][model_name] = t_vanilla_tp['mlp'] * 1e3
        comm_time_data['BTP']['attn'][model_name] = t_btp['attn'] * 1e3
        comm_time_data['BTP']['mlp'][model_name] = t_btp['mlp'] * 1e3
    
    # Generate communication time table
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n⏱️  Communication Time Table (ms)")
        print("=" * 80)
        
        # Print table header
        print(f"{'Communication Time (ms)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
        print("-" * 75)
        
        # Print table rows
        for row_name, mode in table_rows:
            print(f"{row_name:<25} ", end="")
            for model_name in model_order:
                if model_name in comm_time_data[mode]['attn']:
                    total_ms = comm_time_data[mode]['attn'][model_name] + comm_time_data[mode]['mlp'][model_name]
                    print(f"{total_ms:.3f}".ljust(15), end="")
                else:
                    print("N/A".ljust(15), end="")
            print()
    
    # LLaMA 7B specific analysis with different batch sizes
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n📊 LLaMA 7B Communication Analysis (TP=4)")
        print("=" * 80)
        print("Different Batch Sizes")
        print("-" * 80)
    
    llama7b_config = configs["LLaMA-7B"]
    llama7b_rank = int(llama7b_config['hidden_size'] * 0.25)
    batch_sizes = [1, 2, 4, 8]
    llama7b_volume_by_batch = {}
    llama7b_time_by_batch = {}
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"{'Batch Size':<15} {'Fullrank (MB)':<15} {'Vanilla TP (MB)':<15} {'BTP (MB)':<15}")
        print("-" * 60)
    
    for batch_size in batch_sizes:
        # Calculate communication volumes
        fullrank_comm = comm_calculator.calculate_fullrank_communication_volume(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        vanilla_tp_comm = comm_calculator.calculate_vanilla_tp_communication_volume(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            rank=llama7b_rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        btp_comm = comm_calculator.calculate_btp_communication_volume(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            rank=llama7b_rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        fullrank_total_mb = (fullrank_comm['attn'] + fullrank_comm['mlp']) / (1024 * 1024)
        vanilla_tp_total_mb = (vanilla_tp_comm['attn'] + vanilla_tp_comm['mlp']) / (1024 * 1024)
        btp_total_mb = (btp_comm['attn'] + btp_comm['mlp']) / (1024 * 1024)
        llama7b_volume_by_batch[batch_size] = {
            "fullrank_mb": fullrank_total_mb,
            "vanilla_tp_mb": vanilla_tp_total_mb,
            "btp_mb": btp_total_mb,
        }
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{batch_size:<15} {fullrank_total_mb:<15.2f} {vanilla_tp_total_mb:<15.2f} {btp_total_mb:<15.2f}")
    
    # Communication time benchmarking for different batch sizes
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n⏱️  Communication Time Analysis (TP=4)")
        print("=" * 80)
        print("Different Batch Sizes")
        print("-" * 80)
        print(f"{'Batch Size':<15} {'Fullrank (ms)':<15} {'Vanilla TP (ms)':<15} {'BTP (ms)':<15}")
        print("-" * 60)
    
    for batch_size in batch_sizes:
        # Benchmark communication times
        t_fullrank = comm_benchmark.benchmark_fullrank_communication(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        t_vanilla_tp = comm_benchmark.benchmark_vanilla_tp_communication(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            rank=llama7b_rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        t_btp = comm_benchmark.benchmark_btp_communication(
            seq_length=llama7b_config['seq_length'],
            hidden_size=llama7b_config['hidden_size'],
            intermediate_size=llama7b_config['intermediate_size'],
            rank=llama7b_rank,
            batch_size=batch_size,
            tp=tp_fixed
        )
        
        fullrank_total_ms = (t_fullrank['attn'] + t_fullrank['mlp']) * 1e3
        vanilla_tp_total_ms = (t_vanilla_tp['attn'] + t_vanilla_tp['mlp']) * 1e3
        btp_total_ms = (t_btp['attn'] + t_btp['mlp']) * 1e3
        llama7b_time_by_batch[batch_size] = {
            "fullrank_ms": fullrank_total_ms,
            "vanilla_tp_ms": vanilla_tp_total_ms,
            "btp_ms": btp_total_ms,
        }
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{batch_size:<15} {fullrank_total_ms:<15.3f} {vanilla_tp_total_ms:<15.3f} {btp_total_ms:<15.3f}")
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_path = Path(__file__).with_name("communication_eff_tables.txt")
        lines = []

        lines.append("📊 Communication Volume Table (MB)")
        lines.append("=" * 80)
        lines.append(f"{'Communication Volume (MB)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
        lines.append("-" * 75)
        for row_name, mode in table_rows:
            row = f"{row_name:<25} "
            for model_name in model_order:
                if model_name in comm_volume_data[mode]['attn']:
                    total_mb = (comm_volume_data[mode]['attn'][model_name] + comm_volume_data[mode]['mlp'][model_name]) / (1024 * 1024)
                    row += f"{total_mb:.2f}".ljust(15)
                else:
                    row += "N/A".ljust(15)
            lines.append(row)

        lines.append("")
        lines.append("⏱️  Communication Time Table (ms)")
        lines.append("=" * 80)
        lines.append(f"{'Communication Time (ms)':<25} {'3B':<15} {'7B':<15} {'13B':<15}")
        lines.append("-" * 75)
        for row_name, mode in table_rows:
            row = f"{row_name:<25} "
            for model_name in model_order:
                if model_name in comm_time_data[mode]['attn']:
                    total_ms = comm_time_data[mode]['attn'][model_name] + comm_time_data[mode]['mlp'][model_name]
                    row += f"{total_ms:.3f}".ljust(15)
                else:
                    row += "N/A".ljust(15)
            lines.append(row)

        lines.append("")
        lines.append("📊 LLaMA 7B Communication Analysis (TP=4)")
        lines.append("=" * 80)
        lines.append("Different Batch Sizes")
        lines.append("-" * 80)
        lines.append(f"{'Batch Size':<15} {'Fullrank (MB)':<15} {'Vanilla TP (MB)':<15} {'BTP (MB)':<15}")
        lines.append("-" * 60)
        for batch_size in batch_sizes:
            if batch_size in llama7b_volume_by_batch:
                entry = llama7b_volume_by_batch[batch_size]
                lines.append(
                    f"{batch_size:<15} {entry['fullrank_mb']:<15.2f} {entry['vanilla_tp_mb']:<15.2f} {entry['btp_mb']:<15.2f}"
                )
            else:
                lines.append(f"{batch_size:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

        lines.append("")
        lines.append("⏱️  Communication Time Analysis (TP=4)")
        lines.append("=" * 80)
        lines.append("Different Batch Sizes")
        lines.append("-" * 80)
        lines.append(f"{'Batch Size':<15} {'Fullrank (ms)':<15} {'Vanilla TP (ms)':<15} {'BTP (ms)':<15}")
        lines.append("-" * 60)
        for batch_size in batch_sizes:
            if batch_size in llama7b_time_by_batch:
                entry = llama7b_time_by_batch[batch_size]
                lines.append(
                    f"{batch_size:<15} {entry['fullrank_ms']:<15.3f} {entry['vanilla_tp_ms']:<15.3f} {entry['btp_ms']:<15.3f}"
                )
            else:
                lines.append(f"{batch_size:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

        output_path.write_text("\n".join(lines) + "\n")
        print(f"\n📝 Saved summary tables to: {output_path}")
        print("\n✅ Communication analysis complete!")
