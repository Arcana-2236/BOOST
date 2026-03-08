import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as torch_dist

from nanotron.config import ParallelismArgs
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_identity


def _import_cola_modules():
    import sys

    cola_dir = "/home/zhengyangwang/nanotron/examples/cola"
    if cola_dir not in sys.path:
        sys.path.insert(0, cola_dir)
    from config_cola_llama import ColaLlamaConfig  # type: ignore
    from cola_llama import Embedding  # type: ignore

    return ColaLlamaConfig, Embedding


@pytest.mark.parametrize("hidden_size,vocab_size", [(16, 32)])
def test_embedding_hidden_slice_grad_needs_tp_backward_allreduce(hidden_size: int, vocab_size: int):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun --standalone --nproc_per_node=4 -m pytest -q -k embedding_hidden_slice_grad")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        pytest.skip("This test expects WORLD_SIZE=4.")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    ColaLlamaConfig, Embedding = _import_cola_modules()
    config = ColaLlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        max_position_embeddings=32,
        attn_rank=8,
        mlp_rank=8,
        vocab_size=vocab_size,
        tie_word_embeddings=False,
        rope_interleaved=False,
    )
    parallel = ParallelismArgs(
        dp=1,
        pp=1,
        tp=world_size,
        expert_parallel_size=1,
        recompute_layer=False,
        tp_mode="ALL_REDUCE",
        tp_linear_async_communication=False,
    )

    emb_mod = Embedding(tp_pg=torch_dist.group.WORLD, config=config, parallel_config=parallel).to(device=device)
    emb_mod.train()
    with torch.no_grad():
        emb_mod.token_embedding.weight.normal_(mean=0.0, std=0.02)

    seq, batch = 5, 3
    if rank == 0:
        ids = torch.randint(low=0, high=vocab_size, size=(batch, seq), device=device, dtype=torch.long)
        mask = torch.ones(batch, seq, device=device, dtype=torch.bool)
    else:
        ids = torch.empty(batch, seq, device=device, dtype=torch.long)
        mask = torch.empty(batch, seq, device=device, dtype=torch.bool)
    torch_dist.broadcast(ids, src=0)
    torch_dist.broadcast(mask, src=0)

    # Local hidden-chunk objective (the training pattern in cola_llama embedding).
    local_chunk = hidden_size // world_size
    start = rank * local_chunk
    end = (rank + 1) * local_chunk
    g_local = torch.randn(seq, batch, local_chunk, device=device, dtype=torch.float32)

    # Build full objective gradient by gathering local chunks from all TP ranks.
    gathered_chunks = [torch.empty_like(g_local) for _ in range(world_size)]
    torch_dist.all_gather(gathered_chunks, g_local, group=torch_dist.group.WORLD)
    g_full = torch.cat(gathered_chunks, dim=-1)

    # Reference: full objective on full embedding output.
    emb_mod.zero_grad(set_to_none=True)
    out_full = emb_mod.token_embedding(ids.transpose(0, 1))
    loss_ref = (out_full.float() * g_full).sum()
    loss_ref.backward()
    grad_ref = emb_mod.token_embedding.weight.grad.detach().clone().float()

    # Old behavior: local hidden slice objective without backward all-reduce.
    emb_mod.zero_grad(set_to_none=True)
    out_old = emb_mod.token_embedding(ids.transpose(0, 1))
    loss_old = (out_old[:, :, start:end].float() * g_local).sum()
    loss_old.backward()
    grad_old = emb_mod.token_embedding.weight.grad.detach().clone().float()

    # Fixed behavior: inject differentiable identity before hidden slicing.
    emb_mod.zero_grad(set_to_none=True)
    out_new = emb_mod.token_embedding(ids.transpose(0, 1))
    out_new = differentiable_identity(out_new, group=torch_dist.group.WORLD)
    loss_new = (out_new[:, :, start:end].float() * g_local).sum()
    loss_new.backward()
    grad_new = emb_mod.token_embedding.weight.grad.detach().clone().float()

    old_diff = float((grad_old - grad_ref).abs().max().item())
    new_diff = float((grad_new - grad_ref).abs().max().item())

    old_diff_t = torch.tensor(old_diff, device=device, dtype=torch.float32)
    new_diff_t = torch.tensor(new_diff, device=device, dtype=torch.float32)
    torch_dist.all_reduce(old_diff_t, op=torch_dist.ReduceOp.MAX)
    torch_dist.all_reduce(new_diff_t, op=torch_dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"[embedding_slice_grad] old_diff_max={old_diff_t.item():.6e} "
            f"new_diff_max={new_diff_t.item():.6e}"
        )

    # The old path should lose cross-rank hidden contributions.
    assert old_diff > 1e-6, f"Expected old path to mismatch reference; got old_diff={old_diff:.3e}"
    # The fixed path should recover full-gradient behavior.
    assert new_diff < 1e-6, f"Fixed path should match reference; got new_diff={new_diff:.3e}"
