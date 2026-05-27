import torch
import time

def bandwidth_test(size_mb=64, iterations=100):
    nbytes = size_mb * 1024 * 1024
    a = torch.randn(nbytes // 4, device='cuda')
    b = torch.empty_like(a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iterations):
        b.copy_(a)
    torch.cuda.synchronize()
    t1 = time.time()
    gb_transferred = (nbytes * iterations) / (1024**3)
    print(f"Bandwidth: {gb_transferred / (t1 - t0):.2f} GB/s")

if __name__ == '__main__':
    bandwidth_test(64, 200)
