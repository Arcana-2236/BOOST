import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

rank = dist.get_rank()
world = dist.get_world_size()

x = torch.tensor([rank], device="cuda")
dist.all_reduce(x)

print(f"HELLO rank={rank}/{world} node={os.uname().nodename} local_rank={local_rank} allreduce_sum={x.item()}", flush=True)

dist.destroy_process_group()