"""
Custom Inference Benchmark for LLM serving.

For scenarios where genai-bench doesn't fit—custom models, non-standard APIs,
or specialized metrics—this provides fine-grained control over measurement.
"""
import time
import torch
import numpy as np
from collections import defaultdict


class InferenceBenchmark:
    """Benchmark inference with detailed metrics."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = defaultdict(list)
        self._warmed_up = False
    
    def benchmark_single_request(self, prompt, max_tokens=512):
        """
        Benchmark a single inference request.
        
        Returns:
            dict with total_time, tokens_per_second, num_tokens
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Warmup on first request
        if not self._warmed_up:
            _ = self.model.generate(**inputs, max_new_tokens=1)
            torch.cuda.synchronize()
            self._warmed_up = True
        
        # Measure generation
        torch.cuda.synchronize()
        start = time.time()
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        total_time = time.time() - start
        
        num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        
        return {
            'total_time': total_time,
            'tokens_per_second': num_tokens / total_time,
            'num_tokens': num_tokens
        }
    
    def get_statistics(self):
        """Get statistical summary of all recorded metrics."""
        stats = {}
        for key, values in self.metrics.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        return stats


def benchmark_cold_start(model, prompts):
    """
    Measure cold start vs warm performance.
    
    Cold start latency—the time for the first request after model loading—
    can be 10-100x slower than warm requests. This matters for autoscaling:
    if cold starts take 30 seconds but warm requests take 100ms, aggressive
    scale-down policies will cause user-facing latency spikes.
    
    Expected output example:
        cold_time: 2.5s
        warm_mean: 0.15s
        cold_overhead: 2.35s
    
    A 16x cold/warm ratio is typical. Use this data to configure autoscaler
    minimum instances—keep enough warm instances to handle baseline traffic.
    """
    torch.cuda.empty_cache()  # Clear cache to simulate cold start
    
    # First request (cold)
    cold_start = time.time()
    _ = model.generate(prompts[0], max_new_tokens=100)
    torch.cuda.synchronize()
    cold_time = time.time() - cold_start
    
    # Subsequent requests (warm)
    warm_times = []
    for prompt in prompts[1:]:
        start = time.time()
        _ = model.generate(prompt, max_new_tokens=100)
        torch.cuda.synchronize()
        warm_times.append(time.time() - start)
    
    return {
        'cold_time': cold_time,
        'warm_mean': np.mean(warm_times),
        'cold_overhead': cold_time - np.mean(warm_times)
    }


def run_reasoning_step(model, step_input, max_new_tokens=64, do_tool_call=None):
    """
    Measure a single reasoning step with optional tool call.
    
    Separates generation time from tool/API call time for visibility
    into where latency comes from in agentic workflows.
    """
    gen_start = time.time()
    out = model.generate(step_input, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    gen_time = time.time() - gen_start

    tool_time = 0.0
    if do_tool_call:
        t0 = time.time()
        tool_result = do_tool_call()
        tool_time = time.time() - t0

    return gen_time, tool_time, out


def measure_reasoning_session(model, session_steps, do_tool_call_fn=None):
    """
    Measure complete reasoning session with per-step breakdown.
    
    Example output for a 5-step reasoning session:
        Step 1: gen=0.12s, tool=0.45s (API call)
        Step 2: gen=0.15s, tool=0.00s
        Step 3: gen=0.18s, tool=0.82s (database query)
        Step 4: gen=0.14s, tool=0.00s
        Step 5: gen=0.16s, tool=0.00s
        Total: 2.02s (gen: 0.75s, tool: 1.27s)
    
    In this example, 63% of latency comes from external calls, not model
    inference. Optimizing the model won't help—you need to optimize or
    parallelize the tool calls.
    """
    per_step = []
    total = 0.0
    
    for step_input in session_steps:
        gen_t, tool_t, out = run_reasoning_step(
            model, step_input, 
            do_tool_call=(do_tool_call_fn if do_tool_call_fn else None)
        )
        per_step.append({
            'gen_time': gen_t, 
            'tool_time': tool_t, 
            'step_total': gen_t + tool_t
        })
        total += gen_t + tool_t

    times = [s['step_total'] for s in per_step]
    return {
        'per_step': per_step,
        'total': total,
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
