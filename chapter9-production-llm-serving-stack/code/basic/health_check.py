"""
Health Check Implementation for LLM Serving

Monitors endpoint health status with periodic checks.
Integrates with load balancers to route traffic away from unhealthy endpoints.

Usage:
    from health_check import HealthChecker
    
    checker = HealthChecker(endpoints, check_interval=30)
    await checker.start()
"""

import asyncio
import httpx
from typing import List


class HealthChecker:
    """Periodic health checker for service endpoints"""

    def __init__(self, endpoints: List[str], check_interval: int = 30):
        self.endpoints = endpoints
        self.check_interval = check_interval
        self.health_status = {endpoint: True for endpoint in endpoints}
        self.running = False

    async def check_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{endpoint}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def check_all(self):
        """Check all endpoints"""
        tasks = [self.check_endpoint(endpoint) for endpoint in self.endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for endpoint, is_healthy in zip(self.endpoints, results):
            self.health_status[endpoint] = (
                is_healthy if isinstance(is_healthy, bool) else False
            )

    async def start(self):
        """Start health checking loop"""
        self.running = True
        while self.running:
            await self.check_all()
            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop health checking"""
        self.running = False

    def is_healthy(self, endpoint: str) -> bool:
        """Check if endpoint is currently healthy"""
        return self.health_status.get(endpoint, False)


if __name__ == "__main__":

    async def main():
        endpoints = [
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8003",
        ]
        checker = HealthChecker(endpoints, check_interval=5)

        # Run a single check
        await checker.check_all()
        print("Health status:")
        for endpoint, status in checker.health_status.items():
            print(f"  {endpoint}: {'healthy' if status else 'unhealthy'}")

    asyncio.run(main())
