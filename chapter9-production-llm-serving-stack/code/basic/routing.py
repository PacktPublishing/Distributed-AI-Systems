"""
Routing Strategies for Multi-Model LLM Serving

Provides different routing strategies for directing requests to appropriate models:
- Feature-based routing: Route based on request content
- A/B traffic splits: Route based on user hash for consistent assignment
- Dynamic model selection: Route based on load and latency budget

Usage:
    from routing import FeatureBasedRouter, ABRouter, DynamicRouter
"""

import random
import hashlib
from typing import Dict, List


class FeatureBasedRouter:
    """Route requests based on content features"""

    def __init__(self):
        self.routes = {
            "code": "code-llama-7b",
            "chat": "llama-2-7b-chat",
            "summarization": "mistral-7b",
            "default": "llama-2-7b",
        }

    def route(self, request: dict) -> str:
        """Route based on request features"""
        # Check explicit model parameter
        if "model" in request:
            return request["model"]

        # Route based on prompt content
        prompt = request.get("prompt", "").lower()

        if any(keyword in prompt for keyword in ["code", "function", "class", "def"]):
            return self.routes["code"]
        elif any(
            keyword in prompt for keyword in ["hello", "hi", "chat", "conversation"]
        ):
            return self.routes["chat"]
        elif any(keyword in prompt for keyword in ["summarize", "summary", "brief"]):
            return self.routes["summarization"]
        else:
            return self.routes["default"]


class ABRouter:
    """Route requests based on A/B split with consistent hashing"""

    def __init__(self, split_ratio: float = 0.5):
        self.split_ratio = split_ratio
        self.model_a = "llama-2-7b"
        self.model_b = "llama-2-13b"

    def route(self, request: dict) -> str:
        """Route based on A/B split"""
        # Use consistent hashing for same user
        user_id = request.get("user_id", "anonymous")
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)

        # Consistent assignment
        if (hash_value % 100) < (self.split_ratio * 100):
            return self.model_a
        else:
            return self.model_b


class DynamicRouter:
    """Route requests based on load and latency budget"""

    def __init__(self):
        self.models = {
            "llama-2-7b": {
                "endpoint": "http://localhost:8002",
                "latency_budget_ms": 500,
                "cost_per_token": 0.001,
            },
            "llama-2-13b": {
                "endpoint": "http://localhost:8003",
                "latency_budget_ms": 1000,
                "cost_per_token": 0.002,
            },
        }
        self.model_loads = {model: 0 for model in self.models}

    def route(self, request: dict) -> str:
        """Route based on load and latency budget"""
        latency_budget = request.get("latency_budget_ms", 1000)
        cost_sensitive = request.get("cost_sensitive", False)

        # Filter models by latency budget
        available_models = [
            model
            for model, config in self.models.items()
            if config["latency_budget_ms"] <= latency_budget
        ]

        if not available_models:
            # Fallback to fastest
            return min(
                self.models.keys(), key=lambda m: self.models[m]["latency_budget_ms"]
            )

        # Select based on cost or load
        if cost_sensitive:
            return min(available_models, key=lambda m: self.models[m]["cost_per_token"])
        else:
            return min(available_models, key=lambda m: self.model_loads[m])

    def update_load(self, model: str, delta: int):
        """Update model load"""
        if model in self.model_loads:
            self.model_loads[model] += delta


if __name__ == "__main__":
    # Test feature-based routing
    feature_router = FeatureBasedRouter()
    print("Feature-based routing:")
    print(f"  Code request: {feature_router.route({'prompt': 'Write a function to sort'})}")
    print(f"  Chat request: {feature_router.route({'prompt': 'Hello, how are you?'})}")
    print(f"  Summary request: {feature_router.route({'prompt': 'Summarize this article'})}")

    # Test A/B routing
    ab_router = ABRouter(split_ratio=0.5)
    print("\nA/B routing:")
    print(f"  User 'alice': {ab_router.route({'user_id': 'alice'})}")
    print(f"  User 'bob': {ab_router.route({'user_id': 'bob'})}")

    # Test dynamic routing
    dynamic_router = DynamicRouter()
    print("\nDynamic routing:")
    print(f"  Low latency budget: {dynamic_router.route({'latency_budget_ms': 300})}")
    print(f"  Cost sensitive: {dynamic_router.route({'cost_sensitive': True})}")
