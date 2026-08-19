"""Benchmarking utilities for latency, throughput, and memory."""
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

logger = logging.getLogger(__name__)

class PipelineBenchmarker:
    """Benchmark end-to-end inference pipeline."""
    
    def __init__(self, pipeline, model, output_dir: Path):
        self.pipeline = pipeline
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_size(self, obj):
        """Get size of object in bytes."""
        import pickle
        try:
            return len(pickle.dumps(obj))
        except:
            return sys.getsizeof(obj)
            
    def run_benchmarks(self, X_test: pd.DataFrame) -> dict:
        """Run all benchmarks."""
        logger.info("Running performance benchmarks...")
        
        results = {}
        
        # 1. Memory footprint
        pipeline_size_mb = self._get_size(self.pipeline) / (1024 * 1024)
        model_size_mb = self._get_size(self.model) / (1024 * 1024)
        
        results["memory"] = {
            "pipeline_size_mb": float(pipeline_size_mb),
            "model_size_mb": float(model_size_mb),
            "total_size_mb": float(pipeline_size_mb + model_size_mb)
        }
        logger.info(f"Memory: Pipeline={pipeline_size_mb:.2f}MB, Model={model_size_mb:.2f}MB")
        
        # 2. Latency & Throughput for different batch sizes
        batch_sizes = [1, 10, 100, 1000]
        latency_results = {}
        
        # Warmup
        if not X_test.empty:
            warmup_sample = X_test.iloc[[0]].copy()
            _ = self.model.predict_proba(self.pipeline.transform(warmup_sample))
            
        for batch_size in batch_sizes:
            if len(X_test) < batch_size:
                logger.warning(f"Not enough test data for batch size {batch_size}. Skipping.")
                continue
                
            sample_df = X_test.sample(n=batch_size, random_state=42).copy()
            
            # Run 5 iterations to average
            n_iters = 5
            total_time = 0
            
            for _ in range(n_iters):
                start_time = time.perf_counter()
                
                # End-to-end: preprocess + predict
                X_processed = self.pipeline.transform(sample_df)
                _ = self.model.predict_proba(X_processed)
                
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
                
            avg_time_ms = (total_time / n_iters) * 1000
            latency_per_sample_ms = avg_time_ms / batch_size
            throughput_sps = 1000 / latency_per_sample_ms
            
            latency_results[str(batch_size)] = {
                "batch_latency_ms": float(avg_time_ms),
                "per_sample_latency_ms": float(latency_per_sample_ms),
                "throughput_samples_per_sec": float(throughput_sps)
            }
            logger.info(f"Batch {batch_size}: {latency_per_sample_ms:.2f} ms/sample | {throughput_sps:.0f} samples/sec")
            
        results["performance"] = latency_results
        
        # Save to disk
        out_path = self.output_dir / "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
            
        return results
