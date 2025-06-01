from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from contextlib import asynccontextmanager
import uvicorn

class PredictRequest(BaseModel):
    title: str
    summary: str

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

@app.post("/predict")
async def predict(request: PredictRequest):
    text = f"{request.title}\n{request.summary}"
    result = app.state.model(text)
    return {"prediction": result[0]["label"]}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
    