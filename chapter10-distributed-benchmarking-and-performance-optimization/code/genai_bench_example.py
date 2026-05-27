"""
Comprehensive inference benchmarking using genai-bench CLI.

This example shows how to programmatically run genai-bench and analyze results.
Use this when you need to integrate benchmarking into CI/CD pipelines or
automate performance regression testing.

genai-bench is a CLI-based benchmarking tool designed for realistic inference
workload testing. Unlike simple load generators that send identical requests,
it supports configurable traffic patterns that mirror production workloads.
"""
import subprocess
import json
import os
from pathlib import Path


def run_genai_benchmark(
    api_base: str,
    api_key: str,
    model_name: str,
    tokenizer_path: str,
    max_requests: int = 1000,
    max_time_minutes: int = 15,
    concurrency: int = 100,
    traffic_scenario: str = "D(100,100)",
    server_engine: str = "vLLM",
    server_gpu_type: str = "H100"
):
    """
    Run genai-bench benchmark via CLI.
    
    Key parameters:
        - api_backend: Use OpenAI-compatible API format (works with vLLM, SGLang)
        - max_time_per_run: Stop after N minutes even if request limit not reached
        - num_concurrency: Maintain N concurrent requests throughout the test
        - traffic_scenario: D(100,100) = deterministic 100 input/output tokens
    
    Traffic scenarios define the distribution of input and output token lengths:
        - D(100,100): Deterministic—all requests have exactly 100 input/output tokens
        - D(512,512): Longer context scenario
        - I(input_tokens, output_tokens): Image-text input with fixed tokens
        - E(input_tokens): Embedding requests
    """
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "openai",
        "--api-base", api_base,
        "--api-key", api_key,
        "--api-model-name", model_name,
        "--model-tokenizer", tokenizer_path,
        "--task", "text-to-text",
        "--max-time-per-run", str(max_time_minutes),
        "--max-requests-per-run", str(max_requests),
        "--num-concurrency", str(concurrency),
        "--traffic-scenario", traffic_scenario,
        "--server-engine", server_engine,
        "--server-gpu-type", server_gpu_type
    ]
    
    print("Running genai-bench benchmark...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    print("Benchmark completed successfully!")
    return result


def run_benchmark_matrix(
    api_base: str,
    api_key: str,
    model_name: str,
    tokenizer_path: str,
    concurrency_levels: list = [1, 8, 16],
    traffic_scenarios: list = ["D(100,100)", "D(512,512)", "D(2048,2048)"]
):
    """
    Run a matrix of benchmarks across concurrency levels and traffic scenarios.
    
    This reveals critical performance characteristics:
        - Low concurrency + short context: Baseline latency without batching effects
        - High concurrency + short context: How well the system batches requests
        - Any concurrency + long context: Memory pressure and KV cache behavior
    
    Look for non-linear latency increases as context length grows—this indicates
    memory bandwidth bottlenecks.
    """
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "openai",
        "--api-base", api_base,
        "--api-key", api_key,
        "--api-model-name", model_name,
        "--model-tokenizer", tokenizer_path,
        "--task", "text-to-text",
        "--max-time-per-run", "15",
        "--max-requests-per-run", "300",
    ]
    
    # Add all concurrency levels
    for c in concurrency_levels:
        cmd.extend(["--num-concurrency", str(c)])
    
    # Add all traffic scenarios
    for ts in traffic_scenarios:
        cmd.extend(["--traffic-scenario", ts])
    
    cmd.extend(["--server-engine", "vLLM", "--server-gpu-type", "H100"])
    
    print(f"Running {len(concurrency_levels)} x {len(traffic_scenarios)} = "
          f"{len(concurrency_levels) * len(traffic_scenarios)} benchmark configurations")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def analyze_experiment_results(experiment_folder: str):
    """Generate Excel report and plots from experiment results."""
    # Generate Excel report
    subprocess.run([
        "genai-bench", "excel",
        "--experiment-folder", experiment_folder,
        "--excel-name", "benchmark_results",
        "--metric-percentile", "mean"
    ])
    
    # Generate plots
    subprocess.run([
        "genai-bench", "plot",
        "--experiments-folder", experiment_folder,
        "--group-key", "traffic_scenario",
        "--preset", "2x4_default"
    ])


if __name__ == "__main__":
    result = run_genai_benchmark(
        api_base="http://localhost:8000",
        api_key="your-api-key",
        model_name="llama-2-7b-chat",
        tokenizer_path="/path/to/tokenizer",
        max_requests=1000,
        concurrency=100,
        traffic_scenario="D(100,100)"
    )
