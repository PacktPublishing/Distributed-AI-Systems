import torch

def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU {i}: {props.name} ({vram_gb:.1f} GB)")

if __name__ == '__main__':
    check_cuda()
