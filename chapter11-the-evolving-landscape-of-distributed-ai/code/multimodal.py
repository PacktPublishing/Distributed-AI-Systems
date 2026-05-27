"""
Multimodal Distributed Training and Inference

This module provides implementations for:
- Vision-Language Model (VLM) distributed training
- Cross-modal attention with tensor parallelism
- Multimodal batch processing
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from dataclasses import dataclass


@dataclass
class MultimodalBatch:
    """A batch containing multiple modalities."""
    text_tokens: torch.Tensor  # [batch, seq_len]
    images: Optional[torch.Tensor] = None  # [batch, channels, height, width]
    audio: Optional[torch.Tensor] = None  # [batch, time, features]
    video: Optional[torch.Tensor] = None  # [batch, frames, channels, height, width]
    
    def to(self, device: torch.device) -> 'MultimodalBatch':
        return MultimodalBatch(
            text_tokens=self.text_tokens.to(device),
            images=self.images.to(device) if self.images is not None else None,
            audio=self.audio.to(device) if self.audio is not None else None,
            video=self.video.to(device) if self.video is not None else None,
        )


class VisionEncoder(nn.Module):
    """Vision encoder using ViT-style architecture."""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 d_model: int = 768, num_layers: int = 12, num_heads: int = 12):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, 
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, d_model, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.encoder(x)
        x = self.norm(x)
        
        return x  # [B, num_patches + 1, d_model]


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer for fusing different modalities.
    
    Supports tensor parallelism by sharding attention heads across GPUs.
    """
    
    def __init__(self, d_model: int, num_heads: int, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Shard heads across tensor parallel ranks
        assert num_heads % tp_size == 0
        self.local_num_heads = num_heads // tp_size
        self.head_dim = d_model // num_heads
        self.local_d = self.local_num_heads * self.head_dim
        
        # Query from one modality, Key/Value from another
        self.q_proj = nn.Linear(d_model, self.local_d)
        self.k_proj = nn.Linear(d_model, self.local_d)
        self.v_proj = nn.Linear(d_model, self.local_d)
        self.o_proj = nn.Linear(self.local_d, d_model)
    
    def forward(self, query_features: torch.Tensor, kv_features: torch.Tensor,
                process_group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        batch, query_len, _ = query_features.shape
        _, kv_len, _ = kv_features.shape
        
        # Project to local heads
        q = self.q_proj(query_features).view(batch, query_len, self.local_num_heads, self.head_dim)
        k = self.k_proj(kv_features).view(batch, kv_len, self.local_num_heads, self.head_dim)
        v = self.v_proj(kv_features).view(batch, kv_len, self.local_num_heads, self.head_dim)
        
        # Attention
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.contiguous().view(batch, query_len, self.local_d)
        output = self.o_proj(attn_output)
        
        # All-reduce across tensor parallel ranks
        if self.tp_size > 1 and dist.is_initialized():
            dist.all_reduce(output, group=process_group)
        
        return output


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model with distributed training support.
    
    Combines a vision encoder with a language model through
    cross-modal attention layers.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12,
                 num_heads: int = 12, image_size: int = 224,
                 tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.d_model = d_model
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            image_size=image_size, d_model=d_model, 
            num_layers=num_layers // 2, num_heads=num_heads
        )
        
        # Text embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        
        # Cross-modal attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(d_model, num_heads, tp_size, tp_rank)
            for _ in range(num_layers // 4)
        ])
        
        # Language model layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4,
                                                    batch_first=True)
        self.lm_layers = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, batch: MultimodalBatch, 
                process_group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        batch_size, seq_len = batch.text_tokens.shape
        
        # Text embedding
        positions = torch.arange(seq_len, device=batch.text_tokens.device)
        text_features = self.token_embed(batch.text_tokens) + self.pos_embed(positions)
        
        # Vision encoding (if images present)
        if batch.images is not None:
            vision_features = self.vision_encoder(batch.images)
            
            # Cross-modal attention: text attends to vision
            for cross_attn in self.cross_attn_layers:
                text_features = text_features + cross_attn(
                    text_features, vision_features, process_group
                )
        
        # Language modeling
        output = self.lm_layers(text_features)
        output = self.norm(output)
        logits = self.lm_head(output)
        
        return logits


class MultimodalDataParallel:
    """
    Data parallel wrapper for multimodal models.
    
    Handles different modalities that may have different batch sizes
    or require different processing.
    """
    
    def __init__(self, model: nn.Module, world_size: int, rank: int):
        self.model = model
        self.world_size = world_size
        self.rank = rank
    
    def forward(self, batch: MultimodalBatch) -> torch.Tensor:
        """Forward pass with gradient synchronization."""
        output = self.model(batch)
        return output
    
    def sync_gradients(self) -> None:
        """Synchronize gradients across all ranks."""
        if not dist.is_initialized():
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size
    
    @staticmethod
    def scatter_batch(batch: MultimodalBatch, world_size: int, rank: int) -> MultimodalBatch:
        """Scatter a batch across workers."""
        batch_size = batch.text_tokens.shape[0]
        local_batch_size = batch_size // world_size
        start = rank * local_batch_size
        end = start + local_batch_size
        
        return MultimodalBatch(
            text_tokens=batch.text_tokens[start:end],
            images=batch.images[start:end] if batch.images is not None else None,
            audio=batch.audio[start:end] if batch.audio is not None else None,
            video=batch.video[start:end] if batch.video is not None else None,
        )


if __name__ == "__main__":
    # Test VLM
    model = VisionLanguageModel(vocab_size=32000, d_model=256, num_layers=4, num_heads=4)
    
    batch = MultimodalBatch(
        text_tokens=torch.randint(0, 32000, (2, 128)),
        images=torch.randn(2, 3, 224, 224),
    )
    
    output = model(batch)
    print(f"VLM Output shape: {output.shape}")  # [2, 128, 32000]
    
    # Test cross-modal attention
    cross_attn = CrossModalAttention(d_model=256, num_heads=8)
    text_feat = torch.randn(2, 128, 256)
    vision_feat = torch.randn(2, 197, 256)  # 196 patches + 1 CLS
    
    fused = cross_attn(text_feat, vision_feat)
    print(f"Cross-modal attention output: {fused.shape}")  # [2, 128, 256]
