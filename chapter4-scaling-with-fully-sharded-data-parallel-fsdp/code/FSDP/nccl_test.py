import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        x = torch.ones(1, device="cuda")
        print(f"[Rank {rank}] Before all_reduce: {x.item()}")

        dist.all_reduce(x, op=dist.ReduceOp.SUM)

        print(f"[Rank {rank}] After all_reduce: {x.item()} (expected {world_size})")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
