"""
Complete example: train a GPT-style transformer with DDP, mixed precision,
DistributedSampler, and checkpointing. See "Real-World Example: Training a
Transformer with DDP" in Chapter 3.

Launch from the chapter directory:
    torchrun --nproc_per_node=8 code/train_transformer_ddp.py
"""
import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for block in self.transformer_blocks:
            src = block(src)
        return self.fc_out(src)


class DummyDataset(Dataset):
    def __init__(self, size=10000, vocab_size=10000, seq_len=128):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,))


def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def get_dataloader(rank, world_size, batch_size=32, seq_len=128):
    dataset = DummyDataset(seq_len=seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader, sampler


def train():
    rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = TransformerModel(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], bucket_cap_mb=50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    dataloader, sampler = get_dataloader(rank, world_size, batch_size=32)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        model.train()
        for batch_idx, src in enumerate(dataloader):
            src = src.to(device)
            tgt = src[:, 1:]
            src = src[:, :-1]
            optimizer.zero_grad()
            with autocast():
                output = model(src)
                output = output.view(-1, output.size(-1))
                tgt = tgt.reshape(-1)
                loss = criterion(output, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        if rank == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
