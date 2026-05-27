"""
Fault Tolerance and Cost Optimization for LLM Serving

Provides cold start handling, autoscaling, request queuing, and cost optimization.

Usage:
    from fault_tolerance import ModelWarmup, Autoscaler, RequestQueue, CostOptimizedRouter
"""

import time
import asyncio
from typing import List, Dict, Optional


class ModelWarmup:
    """Warmup model to avoid cold start latency"""

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.warmed_up = False

    async def warmup(self):
        """Warmup model with dummy requests"""
        if self.warmed_up:
            return

        print("Warming up model...")
        dummy_prompts = ["warmup"] * 10

        # Run warmup requests
        for prompt in dummy_prompts:
            try:
                await self.model_runner.generate_async([prompt], max_tokens=1)
            except Exception as e:
                print(f"Warmup error: {e}")

        self.warmed_up = True
        print("Model warmed up")


class KeepAliveManager:
    """Keep model warm with periodic requests"""

    def __init__(self, model_runner, keepalive_interval: int = 300):
        self.model_runner = model_runner
        self.keepalive_interval = keepalive_interval
        self.running = False

    async def keepalive_loop(self):
        """Periodic keepalive requests"""
        self.running = True
        while self.running:
            await asyncio.sleep(self.keepalive_interval)
            try:
                await self.model_runner.generate_async(["keepalive"], max_tokens=1)
            except Exception as e:
                print(f"Keepalive error: {e}")

    def start(self):
        """Start keepalive loop"""
        asyncio.create_task(self.keepalive_loop())

    def stop(self):
        """Stop keepalive loop"""
        self.running = False


class Autoscaler:
    """Request-based autoscaling for LLM serving"""

    def __init__(
        self, min_replicas: int = 1, max_replicas: int = 10, target_rps: int = 100
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_rps = target_rps
        self.current_replicas = min_replicas
        self.request_queue = []

    def record_request(self):
        """Record incoming request"""
        self.request_queue.append(time.time())
        # Keep only last minute
        cutoff = time.time() - 60
        self.request_queue = [t for t in self.request_queue if t > cutoff]

    def get_current_rps(self) -> float:
        """Get current requests per second"""
        if not self.request_queue:
            return 0.0
        return len(self.request_queue) / 60.0

    def should_scale_up(self) -> bool:
        """Check if should scale up"""
        current_rps = self.get_current_rps()
        if current_rps > self.target_rps * 1.2:  # 20% over target
            if self.current_replicas < self.max_replicas:
                return True
        return False

    def should_scale_down(self) -> bool:
        """Check if should scale down"""
        current_rps = self.get_current_rps()
        if current_rps < self.target_rps * 0.5:  # 50% under target
            if self.current_replicas > self.min_replicas:
                return True
        return False

    async def scale(self):
        """Scale replicas"""
        if self.should_scale_up():
            self.current_replicas += 1
            await self.add_replica()
        elif self.should_scale_down():
            self.current_replicas -= 1
            await self.remove_replica()

    async def add_replica(self):
        """Add a new replica"""
        print(f"Scaling up to {self.current_replicas} replicas")

    async def remove_replica(self):
        """Remove a replica"""
        print(f"Scaling down to {self.current_replicas} replicas")


class RequestQueue:
    """Request queue with backpressure"""

    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.max_size = max_size

    async def enqueue(self, request: dict) -> bool:
        """Enqueue request, return False if queue is full"""
        try:
            await asyncio.wait_for(self.queue.put(request), timeout=0.1)
            return True
        except asyncio.TimeoutError:
            return False  # Queue full

    async def dequeue(self) -> dict:
        """Dequeue request"""
        return await self.queue.get()

    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()

    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.queue.qsize() >= self.max_size


class WorkerPool:
    """Worker pool for processing requests"""

    def __init__(self, num_workers: int, queue: RequestQueue, model_runner):
        self.num_workers = num_workers
        self.queue = queue
        self.model_runner = model_runner
        self.workers = []

    async def worker(self, worker_id: int):
        """Worker that processes requests"""
        while True:
            try:
                request = await self.queue.dequeue()
                result = await self.model_runner.generate_async([request["prompt"]])
                # Send result back
                await request["response_queue"].put(result)
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    def start(self):
        """Start worker pool"""
        for i in range(self.num_workers):
            worker = asyncio.create_task(self.worker(i))
            self.workers.append(worker)

    def stop(self):
        """Stop worker pool"""
        for worker in self.workers:
            worker.cancel()


class CostOptimizedRouter:
    """Route requests to cost-optimized models"""

    def __init__(self):
        self.models = {
            "llama-2-7b": {"cost_per_token": 0.001, "latency_ms": 100},
            "llama-2-13b": {"cost_per_token": 0.002, "latency_ms": 200},
            "mistral-7b": {"cost_per_token": 0.0015, "latency_ms": 150},
        }

    def route(self, request: dict) -> str:
        """Route to cost-optimized model"""
        cost_sensitive = request.get("cost_sensitive", False)
        latency_budget = request.get("latency_budget_ms", 1000)

        if cost_sensitive:
            # Select cheapest model within latency budget
            available = [
                (model, config)
                for model, config in self.models.items()
                if config["latency_ms"] <= latency_budget
            ]
            if available:
                return min(available, key=lambda x: x[1]["cost_per_token"])[0]

        # Default to fastest
        return min(self.models.items(), key=lambda x: x[1]["latency_ms"])[0]


if __name__ == "__main__":
    # Test autoscaler
    scaler = Autoscaler(min_replicas=1, max_replicas=5, target_rps=100)
    for _ in range(200):
        scaler.record_request()
    print(f"Current RPS: {scaler.get_current_rps():.2f}")
    print(f"Should scale up: {scaler.should_scale_up()}")

    # Test cost-optimized router
    router = CostOptimizedRouter()
    print(f"\nCost-sensitive routing: {router.route({'cost_sensitive': True})}")
    print(f"Low latency routing: {router.route({'latency_budget_ms': 150})}")
