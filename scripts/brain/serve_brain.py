import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import os
import numpy as np
from model import RetailSalesPredictor
import argparse

# New Imports for Memory/RAG
import database
from rag import RAGMemory

app = FastAPI(title="Retail Cognitive Brain", description="Light Deep Learning Service for Retail Analytics")

# Global State
model = None
scaler = None
rag_engine = None

class PredictionRequest(BaseModel):
    day_of_week: int
    is_weekend: int
    is_holiday: int
    promo: int
    rainfall: float
    footfall: int
    inventory: int

class UserProfileRequest(BaseModel):
    user_id: str
    name: str
    email: str
    preferences: dict = None

class MemoryRequest(BaseModel):
    user_id: str
    text: str

class SearchRequest(BaseModel):
    user_id: str = None
    query: str
    k: int = 3

@app.on_event("startup")
async def startup_event():
    global model, scaler, rag_engine
    
    # 0. Init Database
    try:
        database.init_db()
    except Exception as e:
        print(f"!! DB Init Failed: {e}")
    
    # 1. RAG Engine
    rag_engine = RAGMemory()
    
    # 2. Load Scaler
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.json")
    if os.path.exists(scaler_path):
        with open(scaler_path, "r") as f:
            scaler = json.load(f)
        print("--> Scaler loaded.")
    else:
        print("!! Scaler not found. Training might not have run.")
        
    # 3. Load Model
    model_path = os.path.join(os.path.dirname(__file__), "sales_brain.pth")
    input_dim = len(scaler["features"]) if scaler else 7
    model = RetailSalesPredictor(input_dim=input_dim)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("--> Brain Model loaded.")
    else:
        print("!! Model weights not found.")

# --- Prediction Endpoint ---

@app.post("/predict_sales")
async def predict_sales(req: PredictionRequest):
    global model, scaler
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Brain not initialized")
        
    input_data = [req.day_of_week, req.is_weekend, req.is_holiday, req.promo, req.rainfall, req.footfall, req.inventory]
    
    X = np.array(input_data)
    X_mean = np.array(scaler["X_mean"])
    X_std = np.array(scaler["X_std"])
    X_scaled = (X - X_mean) / X_std
    
    with torch.no_grad():
        tensor_in = torch.FloatTensor(X_scaled).unsqueeze(0)
        output_scaled = model(tensor_in).item()
        
    y_mean = scaler["y_mean"]
    y_std = scaler["y_std"]
    prediction = (output_scaled * y_std) + y_mean
    
    return {
        "predicted_sales": max(0, int(prediction)),
        "confidence": "High",
        "model_type": "Residual MLP (Deep Learning)"
    }

# --- User Profile Endpoints ---

@app.post("/user/profile")
async def update_profile(req: UserProfileRequest):
    database.upsert_user(req.user_id, req.name, req.email, req.preferences)
    return {"status": "success", "user_id": req.user_id}

@app.get("/user/profile/{user_id}")
async def get_profile(user_id: str):
    profile = database.get_user(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile

# --- RAG / Memory Endpoints ---

@app.post("/memory/add")
async def add_memory(req: MemoryRequest):
    global rag_engine
    
    # 1. Store in DB (Persistent Log)
    interaction_id = database.add_interaction(req.user_id, req.text)
    if not interaction_id:
        raise HTTPException(status_code=500, detail="Database write failed")
        
    # 2. Add to Vector Store (RAG)
    rag_engine.add_memory(req.user_id, req.text, interaction_id)
    
    return {"status": "stored", "interaction_id": interaction_id}

@app.post("/memory/search")
async def search_memory(req: SearchRequest):
    global rag_engine
    results = rag_engine.search(req.query, user_id=req.user_id, k=req.k)
    return {"query": req.query, "results": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
