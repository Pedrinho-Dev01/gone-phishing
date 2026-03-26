"""
Spam Detection + Emotion Analysis API
Ensemble of RoBERTa-Large + ELECTRA-Large classifiers.
Run with: uvicorn api:app --reload
"""

import json
from typing import Optional

import email
from email import policy as email_policy
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    ElectraForSequenceClassification,
    RobertaForSequenceClassification,
)

# ── Config ────────────────────────────────────────────────────────────────────

ROBERTA_SPAM_REPO   = "Dpedrinho01/trained_roberta_large"
ELECTRA_SPAM_REPO   = "Dpedrinho01/trained_electra_large"
ROBERTA_EMOTION_REPO = "Dpedrinho01/trained_roberta_emotion"
ELECTRA_EMOTION_REPO = "Dpedrinho01/trained_electra_emotion"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAYBE_SPAM_UPPER = 0.50   # [threshold, MAYBE_SPAM_UPPER) → "maybe spam"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Spam Detection + Emotion Analysis API",
    description="Ensemble of RoBERTa-Large + ELECTRA-Large for spam/ham classification and emotion detection.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading ─────────────────────────────────────────────────────────────

class SpamModelBundle:
    """Binary spam/ham classifier with a single threshold."""

    def __init__(self, repo_id: str, model_class):
        print(f"Loading {repo_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = model_class.from_pretrained(repo_id)
        self.model.to(DEVICE)
        self.model.eval()

        from huggingface_hub import hf_hub_download
        threshold_path = hf_hub_download(repo_id=repo_id, filename="threshold_config.json")
        with open(threshold_path) as f:
            cfg = json.load(f)
        self.threshold: float = cfg["recommended_threshold"]
        print(f"  ✓ {repo_id} loaded (threshold={self.threshold}, device={DEVICE})")

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
        logits = self.model(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[0, 1].item()
        return proba


class EmotionModelBundle:
    """Multi-label emotion classifier with per-class thresholds."""

    def __init__(self, repo_id: str, model_class):
        print(f"Loading {repo_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = model_class.from_pretrained(repo_id)
        self.model.to(DEVICE)
        self.model.eval()

        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=repo_id, filename="model_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        self.id2label: dict[str, str] = cfg["id2label"]
        self.threshold_global: float = cfg["threshold_global"]
        self.threshold_per_class: dict[str, float] = cfg["threshold_per_class"]
        self.num_labels: int = cfg["num_labels"]
        print(f"  ✓ {repo_id} loaded ({self.num_labels} emotions, device={DEVICE})")

    @torch.no_grad()
    def predict_proba(self, text: str) -> dict[str, float]:
        """Return {emotion: probability} for all emotion classes."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        # Multi-label → sigmoid
        probas = torch.sigmoid(logits)[0].cpu().tolist()
        return {self.id2label[str(i)]: round(probas[i], 4) for i in range(self.num_labels)}


# Global model instances
roberta_spam_bundle: Optional[SpamModelBundle] = None
electra_spam_bundle: Optional[SpamModelBundle] = None
roberta_emotion_bundle: Optional[EmotionModelBundle] = None
electra_emotion_bundle: Optional[EmotionModelBundle] = None


@app.on_event("startup")
def load_models():
    global roberta_spam_bundle, electra_spam_bundle
    global roberta_emotion_bundle, electra_emotion_bundle

    roberta_spam_bundle = SpamModelBundle(ROBERTA_SPAM_REPO, RobertaForSequenceClassification)
    electra_spam_bundle = SpamModelBundle(ELECTRA_SPAM_REPO, ElectraForSequenceClassification)
    roberta_emotion_bundle = EmotionModelBundle(ROBERTA_EMOTION_REPO, RobertaForSequenceClassification)
    electra_emotion_bundle = EmotionModelBundle(ELECTRA_EMOTION_REPO, ElectraForSequenceClassification)
    print(f"All models ready on {DEVICE}.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    model: str = "ensemble"

class ModelResult(BaseModel):
    spam_probability: float
    is_spam: bool
    threshold: float

class PredictResponse(BaseModel):
    text: str
    model_used: str
    is_spam: bool
    maybe_spam: bool
    spam_probability: float
    ensemble_threshold: float
    maybe_spam_upper_threshold: float
    roberta: Optional[ModelResult] = None
    electra: Optional[ModelResult] = None


class EmotionScore(BaseModel):
    emotion: str
    probability: float
    detected: bool
    threshold: float

class EmotionModelResult(BaseModel):
    emotions: list[EmotionScore]

class EmotionPredictRequest(BaseModel):
    text: str

class EmotionPredictResponse(BaseModel):
    text: str
    detected_emotions: list[str]
    all_scores: list[EmotionScore]  # ensemble averaged, sorted by probability
    roberta: Optional[EmotionModelResult] = None
    electra: Optional[EmotionModelResult] = None


class EmlRequest(BaseModel):
    filename: str
    content: str  # base64 encoded

class FullEmlResponse(BaseModel):
    spam: PredictResponse
    emotion: EmotionPredictResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_spam(proba: float, threshold: float) -> dict:
    maybe_spam = threshold <= proba < MAYBE_SPAM_UPPER
    is_spam    = proba >= MAYBE_SPAM_UPPER
    return {"is_spam": is_spam, "maybe_spam": maybe_spam}


def ensemble_emotions(
    roberta_probas: dict[str, float],
    electra_probas: dict[str, float],
    threshold_per_class: dict[str, float],
) -> tuple[list[str], list[EmotionScore]]:
    """Average both models' probabilities and apply per-class thresholds."""
    all_scores: list[EmotionScore] = []
    detected: list[str] = []

    for emotion, r_prob in roberta_probas.items():
        e_prob = electra_probas.get(emotion, 0.0)
        avg_prob = round((r_prob + e_prob) / 2, 4)
        threshold = threshold_per_class.get(emotion, 0.4)
        is_detected = avg_prob >= threshold
        all_scores.append(EmotionScore(
            emotion=emotion,
            probability=avg_prob,
            detected=is_detected,
            threshold=threshold,
        ))
        if is_detected:
            detected.append(emotion)

    all_scores.sort(key=lambda x: x.probability, reverse=True)
    return detected, all_scores


def _emotion_model_result(bundle: EmotionModelBundle, probas: dict[str, float]) -> EmotionModelResult:
    scores = []
    for emotion, prob in probas.items():
        threshold = bundle.threshold_per_class.get(emotion, bundle.threshold_global)
        scores.append(EmotionScore(
            emotion=emotion,
            probability=prob,
            detected=prob >= threshold,
            threshold=threshold,
        ))
    scores.sort(key=lambda x: x.probability, reverse=True)
    return EmotionModelResult(emotions=scores)


# ── EML parser ────────────────────────────────────────────────────────────────

def extract_text_from_eml(raw_bytes: bytes) -> str:
    msg = email.message_from_bytes(raw_bytes, policy=email_policy.default)
    parts = []

    subject = msg.get("subject", "")
    if subject:
        parts.append(f"Subject: {subject}")

    from_addr = msg.get("from", "")
    if from_addr:
        parts.append(f"From: {from_addr}")

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                parts.append(part.get_content())
            elif ct == "text/html" and "attachment" not in cd and not any("plain" in p for p in parts):
                import html as html_lib, re
                raw_html = part.get_content()
                text = re.sub(r"<[^>]+>", " ", raw_html)
                text = html_lib.unescape(text)
                text = re.sub(r"\s+", " ", text).strip()
                parts.append(text)
    else:
        parts.append(msg.get_content())

    return "\n".join(parts).strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Spam Detection + Emotion Analysis API is running."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "spam_models_loaded": roberta_spam_bundle is not None and electra_spam_bundle is not None,
        "emotion_models_loaded": roberta_emotion_bundle is not None and electra_emotion_bundle is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty.")

    model_key = req.model.lower()
    if model_key not in ("ensemble", "roberta", "electra"):
        raise HTTPException(status_code=422, detail="model must be 'ensemble', 'roberta', or 'electra'.")

    roberta_proba = roberta_spam_bundle.predict_proba(req.text)
    electra_proba = electra_spam_bundle.predict_proba(req.text)

    roberta_result = ModelResult(
        spam_probability=round(roberta_proba, 4),
        is_spam=roberta_proba >= MAYBE_SPAM_UPPER,
        threshold=roberta_spam_bundle.threshold,
    )
    electra_result = ModelResult(
        spam_probability=round(electra_proba, 4),
        is_spam=electra_proba >= MAYBE_SPAM_UPPER,
        threshold=electra_spam_bundle.threshold,
    )

    if model_key == "roberta":
        final_proba = roberta_proba
        ensemble_threshold = roberta_spam_bundle.threshold
    elif model_key == "electra":
        final_proba = electra_proba
        ensemble_threshold = electra_spam_bundle.threshold
    else:
        final_proba = (roberta_proba + electra_proba) / 2
        ensemble_threshold = (roberta_spam_bundle.threshold + electra_spam_bundle.threshold) / 2

    flags = classify_spam(final_proba, ensemble_threshold)

    return PredictResponse(
        text=req.text,
        model_used=model_key,
        is_spam=flags["is_spam"],
        maybe_spam=flags["maybe_spam"],
        spam_probability=round(final_proba, 4),
        ensemble_threshold=ensemble_threshold,
        maybe_spam_upper_threshold=MAYBE_SPAM_UPPER,
        roberta=roberta_result,
        electra=electra_result,
    )


@app.post("/predict/emotion", response_model=EmotionPredictResponse)
def predict_emotion(req: EmotionPredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty.")

    roberta_probas = roberta_emotion_bundle.predict_proba(req.text)
    electra_probas = electra_emotion_bundle.predict_proba(req.text)

    # Use roberta's per-class thresholds (both models share the same config structure)
    detected, all_scores = ensemble_emotions(
        roberta_probas, electra_probas, roberta_emotion_bundle.threshold_per_class
    )

    return EmotionPredictResponse(
        text=req.text,
        detected_emotions=detected,
        all_scores=all_scores,
        roberta=_emotion_model_result(roberta_emotion_bundle, roberta_probas),
        electra=_emotion_model_result(electra_emotion_bundle, electra_probas),
    )


@app.post("/predict/batch")
def predict_batch(texts: list[str], model: str = "ensemble"):
    if len(texts) > 50:
        raise HTTPException(status_code=422, detail="Batch size limit is 50.")
    return [predict(PredictRequest(text=t, model=model)) for t in texts]


@app.post("/predict/eml", response_model=FullEmlResponse)
async def predict_eml(req: EmlRequest):
    if not req.filename.endswith(".eml"):
        raise HTTPException(status_code=422, detail="Only .eml files are accepted.")

    import base64
    raw = base64.b64decode(req.content)

    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")

    try:
        text = extract_text_from_eml(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse .eml: {e}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the .eml file.")

    analyzed_text = text.strip()
    print("\n=== [EMAIL SCAN] Content analyzed ===")
    print(analyzed_text)
    print("=== [END EMAIL CONTENT] ===\n")

    spam_result  = predict(PredictRequest(text=analyzed_text, model="ensemble"))
    emotion_result = predict_emotion(EmotionPredictRequest(text=analyzed_text))

    return FullEmlResponse(spam=spam_result, emotion=emotion_result)