from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from contextlib import asynccontextmanager
from typing import List
import uvicorn
import time

class PredictRequest(BaseModel):
    title: str
    summary: str

class BatchPredictRequest(BaseModel):
    papers: List[PredictRequest]

class PredictResponse(BaseModel):
    prediction: str

class BatchPredictResponse(BaseModel):
    predictions: List[str]
    inference_time_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once on startup
    app.state.model = pipeline(
        "text-classification",
        model="gpham/scibert-finetuned-arxiv-42",
        device=-1,
        max_length=256,
        truncation=True,
        padding=True
    )
    yield
    # Cleanup logic if needed

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single prediction endpoint (keep for backward compatibility)"""
    text = f"{request.title}\n{request.summary}"
    result = app.state.model(text)
    return {"prediction": result[0]["label"]}

@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction endpoint - much more efficient!"""
    start_time = time.time()
    
    # Prepare all texts for batch processing
    texts = [f"{paper.title}\n{paper.summary}" for paper in request.papers]
    
    # Single model call for all papers (MUCH faster!)
    results = app.state.model(texts)
    
    # Extract predictions
    predictions = [result["label"] for result in results]
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    return {
        "predictions": predictions,
        "inference_time_ms": inference_time
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": hasattr(app.state, 'model')}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "ArXiv Classification API",
        "endpoints": {
            "single": "/predict",
            "batch": "/predict_batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)