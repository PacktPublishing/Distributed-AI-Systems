"""
Canary Deployment and A/B Testing for LLM Serving

Provides canary deployment with gradual traffic shifting and automated rollback,
plus A/B testing framework for comparing model performance in production.

Usage:
    from canary import CanaryDeployment, TrafficShifter, ABTestFramework
"""

import random
import hashlib
from typing import Dict, List
from dataclasses import dataclass


class CanaryDeployment:
    """Canary deployment with metrics tracking and automated decisions"""

    def __init__(
        self, stable_model: str, canary_model: str, traffic_percent: float = 0.1
    ):
        self.stable_model = stable_model
        self.canary_model = canary_model
        self.traffic_percent = traffic_percent
        self.metrics = {
            "stable": {"requests": 0, "errors": 0, "latency_sum": 0.0},
            "canary": {"requests": 0, "errors": 0, "latency_sum": 0.0},
        }

    def route(self, request: dict) -> str:
        """Route request to stable or canary"""
        if random.random() < self.traffic_percent:
            return self.canary_model
        else:
            return self.stable_model

    def record_metrics(self, model: str, latency: float, error: bool = False):
        """Record metrics for model"""
        if model in self.metrics:
            self.metrics[model]["requests"] += 1
            self.metrics[model]["latency_sum"] += latency
            if error:
                self.metrics[model]["errors"] += 1

    def get_error_rate(self, model: str) -> float:
        """Get error rate for model"""
        if model not in self.metrics:
            return 0.0
        m = self.metrics[model]
        if m["requests"] == 0:
            return 0.0
        return m["errors"] / m["requests"]

    def get_avg_latency(self, model: str) -> float:
        """Get average latency for model"""
        if model not in self.metrics:
            return 0.0
        m = self.metrics[model]
        if m["requests"] == 0:
            return 0.0
        return m["latency_sum"] / m["requests"]

    def should_promote(self) -> bool:
        """Check if canary should be promoted"""
        canary_error_rate = self.get_error_rate(self.canary_model)
        stable_error_rate = self.get_error_rate(self.stable_model)
        canary_latency = self.get_avg_latency(self.canary_model)
        stable_latency = self.get_avg_latency(self.stable_model)

        # Promote if canary is better or similar
        if canary_error_rate <= stable_error_rate * 1.1:  # Allow 10% tolerance
            if canary_latency <= stable_latency * 1.2:  # Allow 20% latency increase
                return True

        return False

    def should_rollback(self) -> bool:
        """Check if canary should be rolled back"""
        canary_error_rate = self.get_error_rate(self.canary_model)
        stable_error_rate = self.get_error_rate(self.stable_model)

        # Rollback if canary error rate is significantly worse
        if canary_error_rate > stable_error_rate * 2.0:
            return True

        return False


class TrafficShifter:
    """Gradual traffic shifting for canary deployments"""

    def __init__(self, stable_model: str, canary_model: str):
        self.stable_model = stable_model
        self.canary_model = canary_model
        self.canary_percent = 0.0
        self.shift_steps = [0.1, 0.25, 0.5, 0.75, 1.0]  # Gradual steps
        self.current_step = 0

    def route(self, request: dict) -> str:
        """Route based on current traffic percentage"""
        if random.random() < self.canary_percent:
            return self.canary_model
        else:
            return self.stable_model

    def increase_traffic(self) -> bool:
        """Increase canary traffic to next step"""
        if self.current_step < len(self.shift_steps) - 1:
            self.current_step += 1
            self.canary_percent = self.shift_steps[self.current_step]
            return True
        return False

    def decrease_traffic(self):
        """Decrease canary traffic (rollback)"""
        if self.current_step > 0:
            self.current_step -= 1
            self.canary_percent = self.shift_steps[self.current_step]


@dataclass
class ABTestConfig:
    test_name: str
    variants: Dict[str, float]  # variant_name -> traffic_percent
    metrics: List[str]  # Metrics to track


class ABTestFramework:
    """A/B testing framework for comparing model performance"""

    def __init__(self):
        self.tests: Dict[str, ABTestConfig] = {}
        self.results: Dict[str, Dict[str, Dict]] = {}

    def register_test(self, config: ABTestConfig):
        """Register an A/B test"""
        self.tests[config.test_name] = config
        self.results[config.test_name] = {
            variant: {metric: [] for metric in config.metrics}
            for variant in config.variants.keys()
        }

    def assign_variant(self, test_name: str, user_id: str) -> str:
        """Assign user to a variant using consistent hashing"""
        if test_name not in self.tests:
            return "default"

        config = self.tests[test_name]

        # Consistent hashing for same user
        hash_value = int(
            hashlib.md5(f"{test_name}:{user_id}".encode()).hexdigest(), 16
        )
        cumulative = 0.0

        for variant, percent in config.variants.items():
            cumulative += percent
            if (hash_value % 100) < (cumulative * 100):
                return variant

        # Fallback to first variant
        return list(config.variants.keys())[0]

    def record_metric(self, test_name: str, variant: str, metric: str, value: float):
        """Record a metric value"""
        if test_name in self.results:
            if variant in self.results[test_name]:
                if metric in self.results[test_name][variant]:
                    self.results[test_name][variant][metric].append(value)

    def get_results(self, test_name: str) -> Dict:
        """Get test results with mean and count for each metric"""
        if test_name not in self.results:
            return {}

        results = {}
        for variant, metrics in self.results[test_name].items():
            results[variant] = {
                metric: {
                    "mean": sum(values) / len(values) if values else 0,
                    "count": len(values),
                }
                for metric, values in metrics.items()
            }

        return results


if __name__ == "__main__":
    # Test canary deployment
    canary = CanaryDeployment("llama-2-7b", "llama-2-13b", traffic_percent=0.1)
    print("Canary deployment test:")
    counts = {"llama-2-7b": 0, "llama-2-13b": 0}
    for _ in range(1000):
        model = canary.route({})
        counts[model] += 1
    print(f"  Traffic distribution: {counts}")

    # Test A/B framework
    ab = ABTestFramework()
    ab.register_test(
        ABTestConfig(
            test_name="model_comparison",
            variants={"llama-2-7b": 0.5, "mistral-7b": 0.5},
            metrics=["latency", "quality_score"],
        )
    )
    print("\nA/B test assignment:")
    for user in ["alice", "bob", "charlie", "david"]:
        variant = ab.assign_variant("model_comparison", user)
        print(f"  {user}: {variant}")
