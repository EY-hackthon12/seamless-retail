import asyncio
import aiohttp
import json
import os

# --- Configuration ---
HOST_URL = "http://localhost:8000"
PROMPT = """def calculate_total_revenue(orders):
    \"\"\"
    Calculate total revenue from a list of order dictionaries.
    Each order has 'items' (list of dicts with 'price', 'quantity').
    \"\"\"
"""

async def main():
    print(f"--> Connecting to Host at {HOST_URL}...")
    
    async with aiohttp.ClientSession() as session:
        # Check Health
        try:
            async with session.get(f"{HOST_URL}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Status: {data['status']}")
                    print(f"   Backend: {data['backend']}")
                    print(f"   Model: {data['model']}")
                else:
                    print(f"❌ Host not ready ({resp.status})")
                    return
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return

        print("\n--> Testing Streaming Generation...")
        print(f"Prompt: {PROMPT.strip()}...")
        print("-" * 50)
        
        payload = {
            "prompt": PROMPT,
            "max_new_tokens": 128,
            "temperature": 0.2
        }
        
        try:
            async with session.post(f"{HOST_URL}/generate_stream", json=payload) as resp:
                async for chunk in resp.content.iter_any():
                    # Optimized host yields raw bytes/text
                    print(chunk.decode('utf-8', errors='ignore'), end='', flush=True)
            print("\n" + "-" * 50)
            print("✅ Generation Complete!")
            
        except Exception as e:
            print(f"\n❌ Streaming failed: {e}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
