"""Lab 3: serve the `food11@champion` model from the mlflow registry with FastAPI.

    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
    curl -X POST -F "file=@some.jpg" http://127.0.0.1:8000/predict
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import mlflow
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.food11.data import CLASSES  # same alphabetical order ImageFolder used at training time
from src.food11.train import EVAL_TF

MODEL_URI = "models:/food11@champion"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # overridable so the container can point at the host (host.docker.internal)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    # loaded once, not per request; cpu because Lab 2 exported it (pt2) on cpu
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI, model_config={"device": "cpu"})
    yield


app = FastAPI(title="food11", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "file is not an image")
    x = EVAL_TF(img).unsqueeze(0).numpy()  # (1, 3, 224, 224) float32, as logged in Lab 2
    probs = torch.as_tensor(app.state.model.predict(x)).softmax(1)[0]
    conf, idx = probs.max(0)
    return {"category": CLASSES[idx.item()], "confidence": round(conf.item(), 4)}
