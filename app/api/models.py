from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from app.services.model_manager import model_manager

router = APIRouter()

class ModelListResponse(BaseModel):
    available_models: List[str]
    loaded_models: List[str]

class LoadModelRequest(BaseModel):
    model_name: str
    model_type: str = "dummy"  # Options: 'dummy', 'hf', 'llama_cpp'

class PredictRequest(BaseModel):
    model_name: str
    prompt: str
    parameters: Optional[dict] = {}

@router.get("/list", response_model=ModelListResponse)
async def list_models():
    """List all available models in the models/ directory and currently loaded ones."""
    return ModelListResponse(
        available_models=model_manager.list_available_models(),
        loaded_models=list(model_manager.loaded_models.keys())
    )

@router.post("/load")
async def load_model(request: LoadModelRequest):
    """Explicitly load a model into memory."""
    try:
        model_manager.load_model(request.model_name, request.model_type)
        return {"status": "success", "message": f"Model {request.model_name} loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict(request: PredictRequest):
    """Run inference on a specific local model."""
    try:
        response = model_manager.predict(request.model_name, request.prompt, **request.parameters)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
