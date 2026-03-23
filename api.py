"""
Spam Detection API
Ensemble of RoBERTa-Large + ELECTRA-Large classifiers.
Run with: uvicorn api:app --reload
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    ElectraForSequenceClassification,
    RobertaForSequenceClassification,
)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

ROBERTA_DIR = MODELS_DIR / "roberta_large_final"
ELECTRA_DIR = MODELS_DIR / "electra_large_final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Spam Detection API",
    description="Ensemble of RoBERTa-Large + ELECTRA-Large for spam/ham classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading ─────────────────────────────────────────────────────────────

class ModelBundle:
    def __init__(self, model_dir: Path, model_class, tokenizer_class=None):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = model_class.from_pretrained(str(model_dir))
        self.model.to(DEVICE)
        self.model.eval()

        threshold_path = model_dir / "threshold_config.json"
        with open(threshold_path) as f:
            cfg = json.load(f)
        self.threshold: float = cfg["recommended_threshold"]

    @torch.no_grad()
    def predict_proba(self, text: str) -> float:
        """Return P(spam) as a float in [0, 1]."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = self.model(**inputs).logits  # shape (1, 2)
        proba = torch.softmax(logits, dim=-1)[0, 1].item()  # P(class=1 / spam)
        return proba


roberta_bundle: Optional[ModelBundle] = None
electra_bundle: Optional[ModelBundle] = None


@app.on_event("startup")
def load_models():
    global roberta_bundle, electra_bundle
    print("Loading RoBERTa …")
    roberta_bundle = ModelBundle(ROBERTA_DIR, RobertaForSequenceClassification)
    print("Loading ELECTRA …")
    electra_bundle = ModelBundle(ELECTRA_DIR, ElectraForSequenceClassification)
    print(f"Models loaded on {DEVICE}.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    model: str = "ensemble"  # "ensemble" | "roberta" | "electra"

class ModelResult(BaseModel):
    spam_probability: float
    is_spam: bool
    threshold: float

class PredictResponse(BaseModel):
    text: str
    model_used: str
    is_spam: bool
    spam_probability: float
    ensemble_threshold: float
    roberta: Optional[ModelResult] = None
    electra: Optional[ModelResult] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Spam Detection API is running."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": roberta_bundle is not None and electra_bundle is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty.")

    model_key = req.model.lower()
    if model_key not in ("ensemble", "roberta", "electra"):
        raise HTTPException(status_code=422, detail="model must be 'ensemble', 'roberta', or 'electra'.")

    roberta_proba = roberta_bundle.predict_proba(req.text)
    electra_proba = electra_bundle.predict_proba(req.text)

    roberta_result = ModelResult(
        spam_probability=round(roberta_proba, 4),
        is_spam=roberta_proba >= roberta_bundle.threshold,
        threshold=roberta_bundle.threshold,
    )
    electra_result = ModelResult(
        spam_probability=round(electra_proba, 4),
        is_spam=electra_proba >= electra_bundle.threshold,
        threshold=electra_bundle.threshold,
    )

    if model_key == "roberta":
        final_proba = roberta_proba
        ensemble_threshold = roberta_bundle.threshold
    elif model_key == "electra":
        final_proba = electra_proba
        ensemble_threshold = electra_bundle.threshold
    else:
        # Ensemble: average the two probabilities, use average threshold
        final_proba = (roberta_proba + electra_proba) / 2
        ensemble_threshold = (roberta_bundle.threshold + electra_bundle.threshold) / 2

    return PredictResponse(
        text=req.text,
        model_used=model_key,
        is_spam=final_proba >= ensemble_threshold,
        spam_probability=round(final_proba, 4),
        ensemble_threshold=ensemble_threshold,
        roberta=roberta_result,
        electra=electra_result,
    )


@app.post("/predict/batch")
def predict_batch(texts: list[str], model: str = "ensemble"):
    if len(texts) > 50:
        raise HTTPException(status_code=422, detail="Batch size limit is 50.")
    results = []
    for text in texts:
        req = PredictRequest(text=text, model=model)
        results.append(predict(req))
    return results
