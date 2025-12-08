import time
import asyncio
import argparse
import aiohttp
import numpy as np

async def test_endpoint(session, url, prompt, stream=False):
    payload = {
        "prompt": prompt,
        "max_new_tokens": 128,
        "temperature": 0.2,
        "do_sample": True
    }
    
    start_time = time.time()
    ttft = 0
    
    if stream:
        async with session.post(url + "/generate_stream", json=payload) as response:
            first_chunk = True
            async for chunk in response.content.iter_any():
                if first_chunk:
                    ttft = time.time() - start_time
                    first_chunk = False
            total_time = time.time() - start_time
    else:
        async with session.post(url + "/generate", json=payload) as response:
            data = await response.json()
            total_time = time.time() - start_time
            ttft = total_time # For non-streaming, TTFT is total time
            
    return ttft, total_time

async def run_benchmark(url, concurrent_users=10, num_requests=50, stream=False):
    print(f"--> Benchmarking {url} with {concurrent_users} concurrent users, {num_requests} total requests. Stream={stream}")
    
    prompts = [
        "def fibonacci(n):",
        "import pandas as pd\n# Load csv",
        "class NeuralNetwork(nn.Module):",
        "def binary_search(arr, target):",
        "React component for a button:",
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            prompt = prompts[_ % len(prompts)]
            tasks.append(test_endpoint(session, url, prompt, stream))
            
        # Run in chunks of 'concurrent_users'
        results = []
        for i in range(0, len(tasks), concurrent_users):
            chunk = tasks[i:i+concurrent_users]
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
            
    ttfts = [r[0] * 1000 for r in results] # ms
    totals = [r[1] for r in results] # s
    
    print("\n--- Results ---")
    print(f"Avg TTFT: {np.mean(ttfts):.2f} ms")
    print(f"P99 TTFT: {np.percentile(ttfts, 99):.2f} ms")
    print(f"Avg Total Time: {np.mean(totals):.2f} s")
    print(f"Throughput (req/s): {num_requests / sum(totals) * concurrent_users:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.url, args.users, args.requests, args.stream))
