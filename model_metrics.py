import torch
import torch.nn as nn
from thop import profile
import time
from functools import wraps
import numpy as np

def count_flops(model, input_shape, device='cuda'):
    """
    Calculate FLOPs for a single forward pass of the model.
    
    Args:
        model: The UNet model
        input_shape: Tuple of (batch_size, channels, height, width)
        device: Device to run the calculation on
    
    Returns:
        flops: Number of FLOPs
        params: Number of parameters
    """
    model.eval()
    x = torch.randn(input_shape).to(device)
    t = torch.zeros(input_shape[0], dtype=torch.long).to(device)
    
    flops, params = profile(model, inputs=(x, t))
    return flops, params

def measure_inference_time(model, input_shape, num_timesteps, device='cuda', num_runs=10):
    """
    Measure inference time for the full sampling process.
    
    Args:
        model: The UNet model
        input_shape: Tuple of (batch_size, channels, height, width)
        num_timesteps: Number of diffusion timesteps
        device: Device to run the measurement on
        num_runs: Number of runs to average over
    
    Returns:
        avg_time: Average inference time in seconds
        std_time: Standard deviation of inference time
    """
    model.eval()
    times = []
    
    for _ in range(num_runs):
        x = torch.randn(input_shape).to(device)
        t = torch.zeros(input_shape[0], dtype=torch.long).to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(x, t)
        
        # Measure time
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for t in range(num_timesteps):
                t_tensor = torch.tensor([t] * input_shape[0], device=device)
                _ = model(x, t_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def calculate_model_efficiency(model, input_shape, num_timesteps, device='cuda'):
    """
    Calculate comprehensive model efficiency metrics.
    
    Args:
        model: The UNet model
        input_shape: Tuple of (batch_size, channels, height, width)
        num_timesteps: Number of diffusion timesteps
        device: Device to run the calculations on
    
    Returns:
        dict: Dictionary containing efficiency metrics
    """
    # Calculate FLOPs for single forward pass
    flops, params = count_flops(model, input_shape, device)
    
    # Calculate inference time
    avg_time, std_time = measure_inference_time(model, input_shape, num_timesteps, device)
    
    # Calculate total FLOPs for full sampling process
    total_flops = flops * num_timesteps
    
    # Calculate FLOPS (operations per second)
    flops_per_second = total_flops / avg_time
    
    return {
        'parameters': params,
        'flops_per_forward': flops,
        'total_flops': total_flops,
        'inference_time_mean': avg_time,
        'inference_time_std': std_time,
        'flops_per_second': flops_per_second,
        'memory_usage': torch.cuda.max_memory_allocated() / 1024**2  # MB
    }

def print_efficiency_metrics(metrics):
    """Print model efficiency metrics in a readable format."""
    print("\nModel Efficiency Metrics:")
    print(f"Number of Parameters: {metrics['parameters']:,}")
    print(f"FLOPs per Forward Pass: {metrics['flops_per_forward']:,}")
    print(f"Total FLOPs (Full Sampling): {metrics['total_flops']:,}")
    print(f"Average Inference Time: {metrics['inference_time_mean']:.3f} ± {metrics['inference_time_std']:.3f} seconds")
    print(f"FLOPS: {metrics['flops_per_second']:,.2f} ops/second")
    print(f"GPU Memory Usage: {metrics['memory_usage']:.2f} MB") 