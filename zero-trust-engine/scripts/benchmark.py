import time
import asyncio
from httpx import AsyncClient
from src.zta_api import app

async def run_benchmark(num_requests=1000):
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Trigger startup events
        async with client:
            payload = {
                "device_id": "bench_dev",
                "trust_score": 0.8,
                "trust_threshold": 0.5,
                "trust_state": "HIGH",
                "trend": "STABLE",
                "reason": "bench",
                "timestamp": "2026-06-30T12:00:00Z",
                "schema_version": "1.0.0"
            }
            
            start_time = time.time()
            for _ in range(num_requests):
                resp = await client.post("/decision", json=payload)
                assert resp.status_code == 200
            end_time = time.time()
            
            total_time = end_time - start_time
            latency = (total_time / num_requests) * 1000
            throughput = num_requests / total_time
            
            print(f"--- Benchmark: {num_requests} requests ---")
            print(f"Total time: {total_time:.4f} seconds")
            print(f"Latency per request: {latency:.4f} ms")
            print(f"Throughput: {throughput:.2f} req/s")

if __name__ == "__main__":
    asyncio.run(run_benchmark(1000))
