"""
Edge-Cloud Coordination for Distributed Inference

This module provides patterns for hybrid edge-cloud AI deployment:
- Speculative decoding with edge draft and cloud verification
- Intelligent offloading based on latency and confidence
- Request routing between edge and cloud
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol


class Model(Protocol):
    """Protocol for model inference."""
    def generate(self, tokens: list[int], max_new_tokens: int) -> list[int]: ...
    def verify(self, context: list[int], draft: list[int]) -> list[int]: ...
    def infer(self, request: Any) -> Any: ...


class Tokenizer(Protocol):
    """Protocol for tokenizer."""
    eos_token_id: int
    def encode(self, text: str, return_tensors: str = None) -> Any: ...


class EdgeCloudSpeculativeDecoding:
    """
    Speculative decoding with edge-cloud collaboration.
    
    Edge model generates draft tokens quickly, cloud model verifies
    and accepts/rejects them. Achieves 2-3x speedup when draft
    acceptance rate is high.
    """
    
    def __init__(self, edge_model: Model, cloud_model: Model, 
                 tokenizer: Tokenizer, max_draft_tokens: int = 5):
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.tokenizer = tokenizer
        self.max_draft_tokens = max_draft_tokens
        self.eos_token = getattr(tokenizer, 'eos_token_id', 0)
        
        # Statistics
        self.total_drafts = 0
        self.accepted_drafts = 0
    
    def tokenize(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, 'encode'):
            encoded = self.tokenizer.encode(text, return_tensors='pt')
            return encoded[0].tolist() if hasattr(encoded, '__getitem__') else encoded
        return []
    
    def generate(self, prompt: str, max_tokens: int = 100) -> list[int]:
        tokens = self.tokenize(prompt)
        generated = []
        
        while len(generated) < max_tokens:
            draft_tokens = self.edge_model.generate(
                tokens + generated,
                max_new_tokens=self.max_draft_tokens
            )
            
            self.total_drafts += len(draft_tokens)
            
            verified_tokens = self.cloud_model.verify(
                tokens + generated,
                draft_tokens
            )
            
            self.accepted_drafts += len(verified_tokens)
            generated.extend(verified_tokens)
            
            if verified_tokens and verified_tokens[-1] == self.eos_token:
                break
        
        return generated
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_drafts == 0:
            return 0.0
        return self.accepted_drafts / self.total_drafts


class IntelligentOffloading:
    """
    Intelligent routing between edge and cloud inference.
    
    Routes requests based on:
    - Estimated latency requirements
    - Edge model confidence
    - Request complexity
    """
    
    def __init__(self, edge_model: Model, cloud_client: Model, 
                 latency_threshold: float = 100):
        self.edge_model = edge_model
        self.cloud_client = cloud_client
        self.latency_threshold = latency_threshold
        
        # Configurable parameters
        self.edge_base_latency = 50  # ms
        self.cloud_network_latency = 20  # ms
        self.cloud_processing_latency = 30  # ms
        self.confidence_threshold = 0.8
        
        # Statistics
        self.edge_requests = 0
        self.cloud_requests = 0
    
    def route_request(self, request: Any) -> tuple[str, Any]:
        edge_latency = self.estimate_edge_latency(request)
        cloud_latency = self.estimate_cloud_latency(request)
        edge_confidence = self.estimate_edge_confidence(request)
        
        if edge_latency < self.latency_threshold and edge_confidence > self.confidence_threshold:
            self.edge_requests += 1
            return 'edge', self.edge_model.infer(request)
        else:
            self.cloud_requests += 1
            return 'cloud', self.cloud_client.infer(request)
    
    def estimate_edge_latency(self, request: Any) -> float:
        complexity_factor = len(str(request)) / 100
        return self.edge_base_latency * (1 + complexity_factor)
    
    def estimate_cloud_latency(self, request: Any) -> float:
        return self.cloud_network_latency + self.cloud_processing_latency
    
    def estimate_edge_confidence(self, request: Any) -> float:
        # In practice, this would use model uncertainty estimation
        return 0.85
    
    @property
    def edge_ratio(self) -> float:
        total = self.edge_requests + self.cloud_requests
        return self.edge_requests / total if total > 0 else 0.0


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    target: str  # 'edge' or 'cloud'
    latency_estimate: float
    confidence: float
    reason: str


class AdaptiveRouter:
    """
    Adaptive router that learns from past decisions.
    
    Tracks actual vs estimated latency and adjusts routing
    thresholds over time.
    """
    
    def __init__(self, edge_model: Model, cloud_client: Model):
        self.edge_model = edge_model
        self.cloud_client = cloud_client
        
        # Adaptive thresholds
        self.latency_threshold = 100.0
        self.confidence_threshold = 0.8
        
        # Learning rate for threshold adjustment
        self.learning_rate = 0.1
        
        # History for adaptation
        self.edge_latencies: list[float] = []
        self.cloud_latencies: list[float] = []
    
    def route(self, request: Any) -> tuple[RoutingDecision, Any]:
        edge_latency_est = self._estimate_edge_latency(request)
        cloud_latency_est = self._estimate_cloud_latency(request)
        confidence = self._estimate_confidence(request)
        
        if edge_latency_est < cloud_latency_est and confidence > self.confidence_threshold:
            decision = RoutingDecision(
                target='edge',
                latency_estimate=edge_latency_est,
                confidence=confidence,
                reason='Lower latency with sufficient confidence'
            )
            
            start = time.time()
            result = self.edge_model.infer(request)
            actual_latency = (time.time() - start) * 1000
            
            self.edge_latencies.append(actual_latency)
            self._adapt_thresholds(edge_latency_est, actual_latency, 'edge')
        else:
            decision = RoutingDecision(
                target='cloud',
                latency_estimate=cloud_latency_est,
                confidence=confidence,
                reason='Higher accuracy needed or lower cloud latency'
            )
            
            start = time.time()
            result = self.cloud_client.infer(request)
            actual_latency = (time.time() - start) * 1000
            
            self.cloud_latencies.append(actual_latency)
            self._adapt_thresholds(cloud_latency_est, actual_latency, 'cloud')
        
        return decision, result
    
    def _estimate_edge_latency(self, request: Any) -> float:
        if self.edge_latencies:
            return sum(self.edge_latencies[-10:]) / min(len(self.edge_latencies), 10)
        return 50.0
    
    def _estimate_cloud_latency(self, request: Any) -> float:
        if self.cloud_latencies:
            return sum(self.cloud_latencies[-10:]) / min(len(self.cloud_latencies), 10)
        return 70.0
    
    def _estimate_confidence(self, request: Any) -> float:
        return 0.85
    
    def _adapt_thresholds(self, estimated: float, actual: float, target: str):
        error = actual - estimated
        if target == 'edge' and error > 0:
            # Edge was slower than expected, increase threshold
            self.latency_threshold += self.learning_rate * error
        elif target == 'cloud' and error < 0:
            # Cloud was faster than expected, decrease threshold
            self.latency_threshold -= self.learning_rate * abs(error)
