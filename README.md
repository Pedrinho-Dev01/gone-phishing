# Gone Phishing

A spam detection and emotion analysis tool for `.eml` email files, powered by an ensemble of fine-tuned **RoBERTa-Large** and **ELECTRA-Large** models.

## What it does

- **Spam detection** — classifies emails as ham, maybe spam, or spam using an averaged ensemble of two binary classifiers, each with its own calibrated threshold.
- **Emotion analysis** — detects up to 15 emotions (admiration, anger, curiosity, joy, etc.) using a multi-label ensemble with per-class thresholds.
- **Web UI** — drag-and-drop `.eml` files onto the frontend and get instant results with animated probability bars and per-model breakdowns.

## Models

| Model | Repo | Task |
|---|---|---|
| RoBERTa-Large (spam) | `Dpedrinho01/trained_roberta_large` | Binary spam/ham |
| ELECTRA-Large (spam) | `Dpedrinho01/trained_electra_large` | Binary spam/ham |
| RoBERTa-Large (emotion) | `Dpedrinho01/trained_roberta_emotion` | Multi-label emotion |
| ELECTRA-Large (emotion) | `Dpedrinho01/trained_electra_emotion` | Multi-label emotion |

Each model ships a config file (`threshold_config.json` for spam, `model_config.json` for emotion) that is loaded from the Hugging Face Hub at startup.

## API

Run locally with:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model load status and device info |
| `POST` | `/predict` | Spam classification on raw text |
| `POST` | `/predict/emotion` | Emotion detection on raw text |
| `POST` | `/predict/eml` | Full analysis (spam + emotion) from a base64-encoded `.eml` file |
| `POST` | `/predict/batch` | Batch spam classification (up to 50 texts) |

The `/predict/eml` endpoint returns a combined response:

```json
{
  "spam":    { "is_spam": false, "maybe_spam": false, "spam_probability": 0.07, "..." },
  "emotion": { "detected_emotions": ["curiosity", "neutral"], "all_scores": [...] }
}
```

## Project structure

```
├── api.py          # FastAPI backend
├── index.html      # Frontend
├── style.css       # Styles
├── app.js          # Frontend logic
└── requirements.txt
```

## Stack

**Backend** — FastAPI · PyTorch · Hugging Face Transformers  
**Frontend** — Vanilla JS · DM Sans / Playfair Display / DM Mono  
**Hosted inference** — Hugging Face Spaces (`Dpedrinho01/api-host`)