import asyncio
import aiohttp
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass


@dataclass
class CognitiveResponse:
    text: str
    metadata: Dict[str, Any]
    latency_ms: float


class CognitiveBrainClient:
    """Python SDK for the Cognitive Retail Brain API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def health(self) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.get(f"{self.base_url}/health") as resp:
            return await resp.json()
    
    async def chat(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> CognitiveResponse:
        session = await self._get_session()
        payload = {
            "query": message,
            "user_id": user_id,
            "session_id": session_id,
            "context": kwargs
        }
        
        async with session.post(f"{self.base_url}/chat", json=payload) as resp:
            data = await resp.json()
            return CognitiveResponse(
                text=data.get("response", ""),
                metadata=data.get("metadata", {}),
                latency_ms=data.get("latency_ms", 0)
            )
    
    async def chat_stream(
        self,
        message: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        session = await self._get_session()
        payload = {
            "query": message,
            "user_id": user_id,
            "stream": True,
            "context": kwargs
        }
        
        async with session.post(f"{self.base_url}/chat/stream", json=payload) as resp:
            async for line in resp.content:
                if line:
                    text = line.decode("utf-8").strip()
                    if text.startswith("data: "):
                        yield text[6:]
    
    async def predict_sales(
        self,
        day_of_week: int,
        is_weekend: int,
        is_holiday: int,
        promo: int,
        rainfall: float,
        footfall: int,
        inventory: int
    ) -> Dict[str, Any]:
        session = await self._get_session()
        payload = {
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "promo": promo,
            "rainfall": rainfall,
            "footfall": footfall,
            "inventory": inventory
        }
        
        async with session.post(f"{self.base_url}/predict", json=payload) as resp:
            return await resp.json()
    
    async def search_products(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        session = await self._get_session()
        payload = {"query": query, "k": top_k}
        
        async with session.post(f"{self.base_url}/search", json=payload) as resp:
            return await resp.json()


def get_client(base_url: str = "http://localhost:8000") -> CognitiveBrainClient:
    return CognitiveBrainClient(base_url)


async def main():
    client = get_client()
    
    try:
        health = await client.health()
        print(f"Health: {health}")
        
        response = await client.chat("What products are trending this week?")
        print(f"Response: {response.text}")
        print(f"Latency: {response.latency_ms}ms")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
