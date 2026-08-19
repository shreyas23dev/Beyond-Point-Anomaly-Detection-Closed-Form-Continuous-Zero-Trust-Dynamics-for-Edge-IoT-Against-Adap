"""Benchmark script for the Dynamic Trust Engine."""
import time
import os
import psutil
from src.trust_engine import DynamicTrustEngine
from src.trust_models import TrustRequest

def run_benchmark(num_requests=1000):
    print(f"--- Running Benchmark with {num_requests} requests ---")
    
    # Initialize engine
    engine = DynamicTrustEngine(
        config_path="config/trust_config.yaml",
        storage_path="data/benchmark_history.json"
    )
    
    req = TrustRequest(
        device_id="benchmark_device_1",
        device_type="Gateway",
        raw_anomaly_score=0.2,
        anomaly_threshold=0.5,
        confidence=0.9,
        timestamp="2026-06-30T12:00:00Z"
    )
    
    start_time = time.time()
    
    # Run loop
    for _ in range(num_requests):
        engine.process_request(req)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    latency_ms = (total_time / num_requests) * 1000
    throughput = num_requests / total_time
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Latency per request: {latency_ms:.4f} ms")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Memory Usage: {memory_mb:.2f} MB")
    
    # Cleanup
    if os.path.exists("data/benchmark_history.json"):
        os.remove("data/benchmark_history.json")

if __name__ == "__main__":
    run_benchmark()
