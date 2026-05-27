"""
Resource Management and Multi-Tenancy

This module provides implementations for:
- Dynamic GPU allocation
- Multi-tenant GPU scheduling
- Gradient compression for communication efficiency
"""

import time
import torch
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict


@dataclass
class Job:
    """Represents a training or inference job."""
    id: str
    model_size: int  # bytes
    batch_size: int
    priority: int = 1
    submitted_at: float = field(default_factory=time.time)


class AdaptiveGPUAllocator:
    """
    Dynamically allocate GPUs based on workload characteristics.
    
    Estimates GPU requirements based on model size and batch size,
    then allocates available resources or queues the job.
    """
    
    def __init__(self, total_gpus: int, min_gpus_per_job: int = 1,
                 gpu_memory_gb: float = 80.0):
        self.total_gpus = total_gpus
        self.min_gpus_per_job = min_gpus_per_job
        self.gpu_memory_gb = gpu_memory_gb
        self.allocated_gpus: dict[str, int] = {}
        self.job_queue: list[Job] = []
    
    def allocate_for_job(self, job: Job) -> Optional[int]:
        required_gpus = self.estimate_gpu_requirement(job)
        required_gpus = max(required_gpus, self.min_gpus_per_job)
        
        available_gpus = self.total_gpus - sum(self.allocated_gpus.values())
        
        if available_gpus >= required_gpus:
            self.allocated_gpus[job.id] = required_gpus
            return required_gpus
        else:
            self.job_queue.append(job)
            return None
    
    def release(self, job_id: str) -> None:
        if job_id in self.allocated_gpus:
            del self.allocated_gpus[job_id]
            self._process_queue()
    
    def estimate_gpu_requirement(self, job: Job) -> int:
        model_size_gb = job.model_size / (1024 ** 3)
        
        # Memory requirement: model + optimizer states + activations
        # Rough estimate: 4x model size for training
        memory_needed_gb = model_size_gb * 4
        gpus_for_memory = max(1, int(memory_needed_gb / self.gpu_memory_gb) + 1)
        
        # Batch size consideration for throughput
        gpus_for_batch = max(1, job.batch_size // 32)
        
        return min(max(gpus_for_memory, gpus_for_batch), 8)
    
    def _process_queue(self) -> None:
        """Try to allocate GPUs for queued jobs."""
        remaining_queue = []
        for job in sorted(self.job_queue, key=lambda j: (-j.priority, j.submitted_at)):
            if self.allocate_for_job(job) is None:
                remaining_queue.append(job)
        self.job_queue = remaining_queue
    
    @property
    def utilization(self) -> float:
        return sum(self.allocated_gpus.values()) / self.total_gpus


class MultiTenantGPUScheduler:
    """
    Share GPUs across multiple tenants with time slicing.
    
    Uses priority-weighted round-robin scheduling to fairly
    distribute GPU time among tenants.
    """
    
    def __init__(self, num_gpus: int, time_slice_ms: float = 100):
        self.num_gpus = num_gpus
        self.time_slice_ms = time_slice_ms
        self.tenant_queues: dict[str, list[Any]] = {}
        self.tenant_priorities: dict[str, int] = {}
        self.tenant_times: dict[str, float] = {}
        self.current_tenant: Optional[str] = None
    
    def add_tenant(self, tenant_id: str, priority: int = 1) -> None:
        self.tenant_queues[tenant_id] = []
        self.tenant_priorities[tenant_id] = priority
        self.tenant_times[tenant_id] = 0
    
    def remove_tenant(self, tenant_id: str) -> None:
        self.tenant_queues.pop(tenant_id, None)
        self.tenant_priorities.pop(tenant_id, None)
        self.tenant_times.pop(tenant_id, None)
    
    def submit_work(self, tenant_id: str, work: Any) -> bool:
        if tenant_id not in self.tenant_queues:
            return False
        self.tenant_queues[tenant_id].append(work)
        return True
    
    def schedule(self, current_time_ms: float) -> Optional[str]:
        """Select next tenant to run based on priority and fairness."""
        if not self.tenant_queues:
            return None
        
        best_tenant = None
        best_score = -1
        
        for tenant_id, queue in self.tenant_queues.items():
            if not queue:
                continue
            
            priority = self.tenant_priorities[tenant_id]
            time_since_last = current_time_ms - self.tenant_times.get(tenant_id, 0)
            
            # Score combines priority with time since last allocation
            score = priority * (1 + time_since_last / 1000)
            
            if score > best_score:
                best_score = score
                best_tenant = tenant_id
        
        if best_tenant:
            self.current_tenant = best_tenant
            self.tenant_times[best_tenant] = current_time_ms
        
        return best_tenant
    
    def get_stats(self) -> dict:
        total_time = sum(self.tenant_times.values())
        return {
            'num_tenants': len(self.tenant_queues),
            'queue_lengths': {k: len(v) for k, v in self.tenant_queues.items()},
            'time_share': {k: v / total_time if total_time > 0 else 0 
                          for k, v in self.tenant_times.items()},
        }


class GradientCompression:
    """
    Compress gradients to reduce communication overhead.
    
    Supports two methods:
    - Top-k sparsification: Keep only k largest gradients
    - Quantization: Reduce precision to 8-bit
    """
    
    def __init__(self, compression_ratio: float = 0.1, method: str = 'topk'):
        self.compression_ratio = compression_ratio
        self.method = method
        self.residuals: dict[str, torch.Tensor] = {}
    
    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        if self.method == 'topk':
            return self._topk_compress(gradients)
        elif self.method == 'quantization':
            return self._quantize_compress(gradients)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")
    
    def _topk_compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        """Top-k sparsification with error feedback."""
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Add residual from previous iteration (error feedback)
            if name in self.residuals:
                grad = grad + self.residuals[name]
            
            k = max(1, int(grad.numel() * self.compression_ratio))
            flat_grad = grad.flatten()
            
            _, indices = torch.topk(flat_grad.abs(), k)
            values = flat_grad[indices]
            
            # Store residual (unselected gradients)
            mask = torch.zeros_like(flat_grad)
            mask[indices] = 1
            self.residuals[name] = (flat_grad * (1 - mask)).view_as(grad)
            
            compressed[name] = {
                'values': values,
                'indices': indices,
                'shape': grad.shape,
                'original_size': grad.numel(),
                'compressed_size': k,
            }
        
        return compressed
    
    def _quantize_compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        """Quantization to 8-bit."""
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            grad_min = grad.min()
            grad_max = grad.max()
            scale = (grad_max - grad_min) / 255.0
            
            if scale == 0:
                quantized = torch.zeros_like(grad, dtype=torch.uint8)
            else:
                quantized = ((grad - grad_min) / scale).round().to(torch.uint8)
            
            compressed[name] = {
                'quantized': quantized,
                'min': grad_min,
                'max': grad_max,
                'scale': scale,
                'shape': grad.shape,
            }
        
        return compressed
    
    def decompress(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        """Decompress gradients back to full precision."""
        gradients = {}
        for name, comp_data in compressed.items():
            if self.method == 'quantization':
                scale = comp_data['scale']
                grad = comp_data['quantized'].float() * scale + comp_data['min']
                gradients[name] = grad
            else:
                grad = torch.zeros(comp_data['shape'], device=comp_data['values'].device)
                flat_grad = grad.flatten()
                flat_grad[comp_data['indices']] = comp_data['values']
                gradients[name] = grad
        
        return gradients
    
    def get_compression_stats(self, compressed: dict[str, dict]) -> dict:
        """Get compression statistics."""
        if self.method == 'topk':
            total_original = sum(c['original_size'] for c in compressed.values())
            total_compressed = sum(c['compressed_size'] for c in compressed.values())
            ratio = total_original / total_compressed if total_compressed > 0 else 0
        else:
            # Quantization: 4x compression (32-bit to 8-bit)
            ratio = 4.0
        
        return {
            'method': self.method,
            'compression_ratio': ratio,
            'bandwidth_reduction': 1 - 1/ratio if ratio > 0 else 0,
        }


if __name__ == "__main__":
    # Test GPU allocator
    allocator = AdaptiveGPUAllocator(total_gpus=8)
    
    job1 = Job(id="job1", model_size=70 * 1024**3, batch_size=64)  # 70GB model
    job2 = Job(id="job2", model_size=7 * 1024**3, batch_size=32)   # 7GB model
    
    gpus1 = allocator.allocate_for_job(job1)
    gpus2 = allocator.allocate_for_job(job2)
    
    print(f"Job 1 allocated: {gpus1} GPUs")
    print(f"Job 2 allocated: {gpus2} GPUs")
    print(f"Utilization: {allocator.utilization:.1%}")
    
    # Test gradient compression
    compressor = GradientCompression(compression_ratio=0.01, method='topk')
    
    gradients = {
        'layer1.weight': torch.randn(1024, 1024),
        'layer2.weight': torch.randn(1024, 1024),
    }
    
    compressed = compressor.compress(gradients)
    stats = compressor.get_compression_stats(compressed)
    
    print(f"\nGradient Compression:")
    print(f"  Method: {stats['method']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Bandwidth reduction: {stats['bandwidth_reduction']:.1%}")
