import torch
import torch.distributed as dist

def allreduce_microbench(tensor_size=1024*1024, iters=100):
    t = torch.randn(tensor_size, device='cuda')
    for _ in range(iters):
        dist.all_reduce(t)

# NOTE: Requires an initialized process group (NCCL backend).
if __name__ == '__main__':
    # Example usage: initialize process group externally (torchrun)
    allreduce_microbench()
