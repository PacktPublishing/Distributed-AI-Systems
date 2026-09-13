import torch
import time

def bandwidth_test(size_mb=64, iterations=100):
    nbytes = size_mb * 1024 * 1024
    a = torch.randn(nbytes // 4, device='cuda')
    b = torch.empty_like(a)
    # Warmup: first iterations pay JIT/allocation costs and skew the timing.
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iterations):
        b.copy_(a)
    torch.cuda.synchronize()
    t1 = time.time()
    # Each copy_ reads `a` and writes `b`: count 2x the tensor size per iteration.
    gb_transferred = (2 * nbytes * iterations) / (1024**3)
    print(f"Effective bandwidth (read+write): {gb_transferred / (t1 - t0):.2f} GB/s")

if __name__ == '__main__':
    bandwidth_test(64, 200)
