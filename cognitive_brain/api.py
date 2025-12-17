import asyncio
import logging
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from cognitive_brain.core.hardware_detector import HardwareDetector
from cognitive_brain.orchestration.brain_graph import process_query, CognitiveBrain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cognitive Retail Brain",
    description="Deep learning powered retail AI system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

brain: Optional[CognitiveBrain] = None
detector: Optional[HardwareDetector] = None


class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


class PredictRequest(BaseModel):
    day_of_week: int
    is_weekend: int
    is_holiday: int
    promo: int
    rainfall: float
    footfall: int
    inventory: int


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.on_event("startup")
async def startup():
    global brain, detector
    
    logger.info("Initializing Cognitive Brain...")
    detector = HardwareDetector()
    detector.print_summary()
    
    brain = CognitiveBrain()
    logger.info("Brain initialized")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "brain_ready": brain is not None,
        "hardware": detector.get_tier(detector.detect()).name if detector else "unknown"
    }


@app.get("/hardware")
async def hardware_info():
    if not detector:
        raise HTTPException(503, "Detector not initialized")
    
    profile = detector.detect()
    config = detector.get_recommended_config()
    
    return {
        "gpus": [{
            "id": g.device_id,
            "name": g.name,
            "vram_gb": g.total_vram_gb
        } for g in profile.gpus],
        "tier": config.tier.name,
        "backend": config.backend.value,
        "quantization": config.quantization.value
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if not brain:
        raise HTTPException(503, "Brain not initialized")
    
    start = time.perf_counter()
    
    result = await process_query(
        query=request.query,
        user_id=request.user_id,
        session_id=request.session_id,
        context=request.context
    )
    
    latency = (time.perf_counter() - start) * 1000
    
    return {
        "response": result["response"],
        "metadata": result["metadata"],
        "routing": result["routing"],
        "latency_ms": round(latency, 2)
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not brain:
        raise HTTPException(503, "Brain not initialized")
    
    async def generate():
        result = await process_query(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            context=request.context
        )
        
        response = result["response"]
        for i in range(0, len(response), 20):
            chunk = response[i:i+20]
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.02)
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/predict")
async def predict_sales(request: PredictRequest):
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8001/predict_sales",
                json=request.dict()
            ) as resp:
                return await resp.json()
    except Exception as e:
        raise HTTPException(503, f"Brain service unavailable: {e}")


@app.post("/search")
async def search_products(request: SearchRequest):
    return {
        "query": request.query,
        "results": [],
        "total": 0
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
