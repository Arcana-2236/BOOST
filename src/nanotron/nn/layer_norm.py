import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.parallel.sharded_parameters import (
    SplitConfig,
    mark_all_parameters_in_module_as_sharded,
)
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    differentiable_all_reduce_sum,
    differentiable_identity,
)


class TritonLayerNorm(nn.LayerNorm):
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn

        return layer_norm_fn(
            input,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            return_dropout_mask=return_dropout_mask,
        )


# This is equivalent to LLaMA RMSNorm
# https://github.com/huggingface/transformers/blob/28952248b19db29ca25ccf34a5eec413376494a9/src/transformers/models/llama/modeling_llama.py#L112
class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn

        return layer_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )


class DelayedTritonRMSNorm(nn.Module):
    def __init__(
        self, 
        hidden_size,
        pg: dist.ProcessGroup,
        eps=1e-5, 
        device=None, 
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.pg = pg
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

        mark_all_parameters_in_module_as_sharded(
            self,
            pg=self.pg,
            split_config=SplitConfig(split_dim=0),
        )

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn

        # rstd = None
        s_local = (input.float() * input.float()).sum(dim=-1, keepdim=True)  # [*,1], fp32 accumulate
        
        return layer_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        ), s_local


class SyncRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        pg: dist.ProcessGroup,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.pg = pg
        self.world_size = pg.size()
        if hidden_size % self.world_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by TP size ({self.world_size}) for SyncRMSNorm."
            )
        self.hidden_size = hidden_size
        self.hidden_size_local = hidden_size // self.world_size
        self.weight = torch.nn.Parameter(torch.empty(self.hidden_size_local, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

        mark_all_parameters_in_module_as_sharded(
            self,
            pg=self.pg,
            split_config=SplitConfig(split_dim=0),
        )

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x_local: torch.Tensor, return_stats: bool = False):
        x_fp32 = x_local.float()
        weight_fp32 = self.weight.float()

        s_local_fp32 = (x_fp32 * x_fp32).sum(dim=-1, keepdim=True)
        # differentiable_all_reduce_sum has identity backward in Nanotron.
        # Wrap with differentiable_identity so backward contributes the expected
        # all-reduce for exact global-stats gradients.
        s_global_fp32 = differentiable_all_reduce_sum(s_local_fp32.clone(), group=self.pg)
        s_global_fp32 = differentiable_identity(s_global_fp32, group=self.pg)

        d_local = x_local.shape[-1]
        d_full = d_local * self.pg.size()
        rstd_fp32 = torch.rsqrt(s_global_fp32 / d_full + self.eps)
        y_fp32 = x_fp32 * rstd_fp32 * weight_fp32
        y_local = y_fp32.to(x_local.dtype)

        if return_stats:
            return y_local, s_local_fp32, s_global_fp32
        return y_local
