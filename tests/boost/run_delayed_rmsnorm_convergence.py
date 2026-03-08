import torch
from nanotron.nn.layer_norm import DelayedTritonRMSNorm, TritonRMSNorm


class ReferenceRMSNorm(torch.nn.Module):
    """Reference RMSNorm implementation for comparison."""
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, input):
        # Standard RMSNorm: output = (input / rms) * weight
        # where rms = sqrt(mean(input^2) + eps)
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        output = input / rms * self.weight
        return output


if __name__ == "__main__":
    BATCH_SIZE = 1
    SEQ_LEN = 2
    DEVICE, DTYPE = torch.device("cuda:0"), torch.float32
    HIDDEN_SIZE = 1024
    NUM_STEPS = 10_000

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    # Reference RMSNorm (standard implementation)
    ref_rmsnorm = ReferenceRMSNorm(hidden_size=HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    
    # Triton RMSNorm (for comparison)
    triton_rmsnorm = TritonRMSNorm(hidden_size=HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    
    # Delayed Triton RMSNorm (the one we're testing)
    delayed_rmsnorm = DelayedTritonRMSNorm(hidden_size=HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    
    # Initialize weights to be the same
    with torch.no_grad():
        delayed_rmsnorm.weight.copy_(ref_rmsnorm.weight)
        triton_rmsnorm.weight.copy_(ref_rmsnorm.weight)

    ref_optim = torch.optim.Adam(ref_rmsnorm.parameters(), lr=0.1)
    triton_optim = torch.optim.Adam(triton_rmsnorm.parameters(), lr=0.1)
    delayed_optim = torch.optim.Adam(delayed_rmsnorm.parameters(), lr=0.1)

    def loss_function(x):
        return x.sum()

    for step in range(NUM_STEPS):
        # NOTE: just make the output fluctuate a bit
        random = torch.randn(1, device=DEVICE) * 0.01
        
        # Reference RMSNorm
        ref_outputs = ref_rmsnorm(inputs) * random
        
        # Triton RMSNorm
        triton_outputs = triton_rmsnorm(inputs) * random
        
        # Delayed Triton RMSNorm (returns output, s_local)
        delayed_outputs, s_local = delayed_rmsnorm(inputs)
        delayed_outputs = delayed_outputs * random

        ref_loss = loss_function(ref_outputs)
        triton_loss = loss_function(triton_outputs)
        delayed_loss = loss_function(delayed_outputs)

        # Optimize reference
        ref_optim.zero_grad()
        ref_loss.backward()
        ref_optim.step()

        # Optimize triton
        triton_optim.zero_grad()
        triton_loss.backward()
        triton_optim.step()

        # Optimize delayed
        delayed_optim.zero_grad()
        delayed_loss.backward()
        delayed_optim.step()

        if step % 100 == 0:
            print(f"Step: {step}")
            print(f"  ref_outputs sum: {ref_outputs.sum().item():.6f}")
            print(f"  triton_outputs sum: {triton_outputs.sum().item():.6f}")
            print(f"  delayed_outputs sum: {delayed_outputs.sum().item():.6f}")
            print(f"  ref_loss: {ref_loss.item():.6f}, triton_loss: {triton_loss.item():.6f}, delayed_loss: {delayed_loss.item():.6f}")
            print(f"  s_local shape: {s_local.shape}, s_local sum: {s_local.sum().item():.6f}")
            
            # Log output differences
            ref_triton_diff = (ref_outputs - triton_outputs).abs().mean().item()
            ref_delayed_diff = (ref_outputs - delayed_outputs).abs().mean().item()
            print(f"  ref_triton_diff: {ref_triton_diff:.6f}, ref_delayed_diff: {ref_delayed_diff:.6f}")
            print()

