"""
Spam Detection API
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

ROBERTA_REPO = "Dpedrinho01/trained_roberta_large"
ELECTRA_REPO = "Dpedrinho01/trained_electra_large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAYBE_SPAM_UPPER = 0.50   # [threshold, MAYBE_SPAM_UPPER) → "maybe spam"


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
    def __init__(self, repo_id: str, model_class):
        print(f"Loading {repo_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = model_class.from_pretrained(repo_id)
        self.model.to(DEVICE)
        self.model.eval()

        # Load threshold from the repo's threshold_config.json
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
        logits = self.model(**inputs).logits  # shape (1, 2)
        proba = torch.softmax(logits, dim=-1)[0, 1].item()  # P(class=1 / spam)
        return proba


roberta_bundle: Optional[ModelBundle] = None
electra_bundle: Optional[ModelBundle] = None


@app.on_event("startup")
def load_models():
    global roberta_bundle, electra_bundle
    roberta_bundle = ModelBundle(ROBERTA_REPO, RobertaForSequenceClassification)
    electra_bundle = ModelBundle(ELECTRA_REPO, ElectraForSequenceClassification)
    print(f"All models ready on {DEVICE}.")


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
    maybe_spam: bool
    spam_probability: float
    ensemble_threshold: float
    maybe_spam_upper_threshold: float
    roberta: Optional[ModelResult] = None
    electra: Optional[ModelResult] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify(proba: float, threshold: float) -> dict:
    """Return is_spam and maybe_spam flags for a given probability."""
    maybe_spam = threshold <= proba < MAYBE_SPAM_UPPER
    is_spam    = proba >= MAYBE_SPAM_UPPER
    return {"is_spam": is_spam, "maybe_spam": maybe_spam}


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
        is_spam=roberta_proba >= MAYBE_SPAM_UPPER,
        threshold=roberta_bundle.threshold,
    )
    electra_result = ModelResult(
        spam_probability=round(electra_proba, 4),
        is_spam=electra_proba >= MAYBE_SPAM_UPPER,
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

    flags = classify(final_proba, ensemble_threshold)

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


@app.post("/predict/batch")
def predict_batch(texts: list[str], model: str = "ensemble"):
    if len(texts) > 50:
        raise HTTPException(status_code=422, detail="Batch size limit is 50.")
    results = []
    for text in texts:
        req = PredictRequest(text=text, model=model)
        results.append(predict(req))
    return results


# ── EML helper ────────────────────────────────────────────────────────────────

def extract_text_from_eml(raw_bytes: bytes) -> str:
    """Parse a .eml file and return a single string with subject + body text."""
    msg = email.message_from_bytes(raw_bytes, policy=email_policy.default)

    parts = []

    # Subject line
    subject = msg.get("subject", "")
    if subject:
        parts.append(f"Subject: {subject}")

    # From for extra signal
    from_addr = msg.get("from", "")
    if from_addr:
        parts.append(f"From: {from_addr}")

    # Walk MIME parts for text content
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                parts.append(part.get_content())
            elif ct == "text/html" and "attachment" not in cd and not any(p.startswith("Subject") or "plain" in p for p in parts):
                # Fallback to HTML only if no plain text found
                import html as html_lib
                raw_html = part.get_content()
                import re
                text = re.sub(r"<[^>]+>", " ", raw_html)
                text = html_lib.unescape(text)
                text = re.sub(r"\s+", " ", text).strip()
                parts.append(text)
    else:
        parts.append(msg.get_content())

    return "\n".join(parts).strip()

class EmlRequest(BaseModel):
    filename: str
    content: str  # base64 encoded

@app.post("/predict/eml", response_model=PredictResponse)
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

    return predict(PredictRequest(text=analyzed_text, model="ensemble"))