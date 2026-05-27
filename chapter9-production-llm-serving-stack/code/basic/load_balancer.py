"""
Load Balancing Strategies for LLM Serving

Provides different load balancing algorithms:
- Round-robin: Distribute requests evenly across endpoints
- Least connections: Route to endpoint with fewest active connections
- Weighted: Route based on endpoint weights/capacity

Usage:
    from load_balancer import RoundRobinBalancer, LeastConnectionsBalancer, WeightedBalancer
"""

import asyncio
import random
from collections import deque
from typing import List


class RoundRobinBalancer:
    """Distribute requests evenly across endpoints in round-robin fashion"""

    def __init__(self, endpoints: List[str]):
        self.endpoints = deque(endpoints)
        self.health_status = {endpoint: True for endpoint in endpoints}

    def get_endpoint(self) -> str:
        """Get next endpoint in round-robin fashion"""
        attempts = 0
        while attempts < len(self.endpoints):
            endpoint = self.endpoints[0]
            self.endpoints.rotate(1)

            if self.health_status.get(endpoint, False):
                return endpoint

            attempts += 1

        # All unhealthy, return first anyway
        return self.endpoints[0]

    def mark_unhealthy(self, endpoint: str):
        """Mark endpoint as unhealthy"""
        self.health_status[endpoint] = False

    def mark_healthy(self, endpoint: str):
        """Mark endpoint as healthy"""
        self.health_status[endpoint] = True


class LeastConnectionsBalancer:
    """Route to endpoint with fewest active connections"""

    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.connection_counts = {endpoint: 0 for endpoint in endpoints}
        self.lock = asyncio.Lock()

    async def get_endpoint(self) -> str:
        """Get endpoint with least connections"""
        async with self.lock:
            endpoint = min(self.endpoints, key=lambda e: self.connection_counts[e])
            self.connection_counts[endpoint] += 1
            return endpoint

    async def release_endpoint(self, endpoint: str):
        """Release connection from endpoint"""
        async with self.lock:
            if endpoint in self.connection_counts:
                self.connection_counts[endpoint] = max(
                    0, self.connection_counts[endpoint] - 1
                )


class WeightedBalancer:
    """Route based on endpoint weights/capacity"""

    def __init__(self, endpoints: List[tuple]):
        """
        endpoints: List of (endpoint, weight) tuples
        """
        self.endpoints = endpoints
        self.total_weight = sum(weight for _, weight in endpoints)

    def get_endpoint(self) -> str:
        """Get endpoint based on weights"""
        r = random.uniform(0, self.total_weight)
        cumulative = 0

        for endpoint, weight in self.endpoints:
            cumulative += weight
            if r <= cumulative:
                return endpoint

        # Fallback to last endpoint
        return self.endpoints[-1][0]


if __name__ == "__main__":
    # Test round-robin
    endpoints = ["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"]
    rr_balancer = RoundRobinBalancer(endpoints)
    print("Round-robin balancing:")
    for i in range(6):
        print(f"  Request {i+1}: {rr_balancer.get_endpoint()}")

    # Test weighted
    weighted_endpoints = [
        ("http://localhost:8001", 1),
        ("http://localhost:8002", 2),
        ("http://localhost:8003", 3),
    ]
    weighted_balancer = WeightedBalancer(weighted_endpoints)
    print("\nWeighted balancing (1000 requests):")
    counts = {}
    for _ in range(1000):
        ep = weighted_balancer.get_endpoint()
        counts[ep] = counts.get(ep, 0) + 1
    for ep, count in sorted(counts.items()):
        print(f"  {ep}: {count} requests ({count/10:.1f}%)")
