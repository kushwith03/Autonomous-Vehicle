import time
import numpy as np
import torch

def benchmark_inference(predictor, n_runs=200, warmup=20, img_size=(128, 128)):
    """
    Generate random uint8 numpy image of img_size
    Run warmup iterations (not timed)
    Run n_runs iterations, timing each with time.perf_counter()
    Return dict with mean_ms, p50_ms, etc.
    """
    # Generate random uint8 numpy image
    random_img = np.random.randint(0, 256, (img_size[0], img_size[1], 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(warmup):
        _ = predictor.predict(random_img)
        
    latencies = []
    for _ in range(n_runs):
        _ = predictor.predict(random_img)
        latencies.append(predictor.last_latency_ms)
        
    latencies = np.array(latencies)
    
    results = {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies))
    }
    
    return results

def print_benchmark_report(results: dict):
    """
    Prints a formatted table.
    """
    print("\n" + "="*40)
    print(f"{'Metric':<20} | {'Value (ms)':<10}")
    print("-" * 40)
    for k, v in results.items():
        print(f"{k:<20} | {v:<10.2f}")
    print("="*40 + "\n")

def assert_latency_target(results: dict, target_ms: float = 50.0):
    """
    Raises AssertionError with message if mean_ms > target_ms
    """
    mean_ms = results["mean_ms"]
    if mean_ms > target_ms:
        raise AssertionError(f"[FAIL] Mean latency {mean_ms:.2f}ms exceeds target {target_ms:.2f}ms")
