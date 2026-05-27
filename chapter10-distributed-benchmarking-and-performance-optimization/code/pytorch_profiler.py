"""
PyTorch Profiler examples for training performance analysis.

The PyTorch profiler captures both CPU and CUDA operations, giving you a
complete picture of where time is spent. Label code regions with record_function
to identify bottlenecks by phase.
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity


def profile_training_step(model, inputs, targets, criterion, optimizer):
    """
    Profile a single training step with phase breakdown.
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        targets: Target tensor
        criterion: Loss function
        optimizer: Optimizer
    
    Configuration options:
        - activities: Specify CPU and CUDA to capture both host and device operations
        - record_shapes: Records tensor shapes, useful for understanding memory patterns
        - profile_memory: Tracks memory allocations and deallocations
        - with_stack: Captures Python call stacks for deeper debugging
    
    The output table shows operations sorted by total CUDA time. Look for
    operations consuming disproportionate time—these are your optimization targets.
    The exported trace.json can be viewed in chrome://tracing for detailed
    timeline analysis with visual swimlanes for each thread and CUDA stream.
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("forward"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        with record_function("backward"):
            loss.backward()
        
        with record_function("optimizer"):
            optimizer.step()
    
    # Print results sorted by CUDA time
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    # Export for visualization in chrome://tracing
    prof.export_chrome_trace("trace.json")
    
    return prof


def profile_with_schedule(model, dataloader, train_step_fn):
    """
    Profile multiple iterations with automatic warmup handling.
    
    Schedule parameters control the profiling lifecycle:
        - wait=1: Skip the first iteration entirely (cold start)
        - warmup=1: Run one iteration to warm up caches without recording
        - active=3: Actually profile the next 3 iterations
        - repeat=2: Repeat this wait→warmup→active cycle twice
    
    With these settings, iterations 0-1 are skipped/warmup, 2-4 are profiled,
    5-6 are skipped/warmup again, and 7-9 are profiled.
    
    The tensorboard_trace_handler automatically saves traces to ./log/
    for visualization with: tensorboard --logdir=./log
    """
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,      # Skip first iteration
            warmup=1,    # Warmup for 1 iteration
            active=3,    # Profile 3 iterations
            repeat=2     # Repeat schedule 2 times
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, data in enumerate(dataloader):
            prof.step()
            train_step_fn(model, data)
    
    return prof
