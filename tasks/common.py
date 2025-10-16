# tasks/common.py
import torch.distributed as dist

def init_dist(rank, world_size, backend="gloo"):
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
