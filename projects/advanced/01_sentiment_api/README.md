# Sentiment Analysis API

**Difficulty:** Advanced
**Status:** Planned
**Estimated time:** 8–10 hours

---

## The Business Problem

An e-commerce company receives 10,000 product reviews per day.
Their team manually reads and categorises them — which takes 3 people full-time.

The goal: build a REST API that classifies any review as **positive, neutral, or negative** in under 200ms, so the company can automate their review pipeline.

This project demonstrates the full path from a trained notebook model to a deployed, callable service.

---

## What makes this "advanced"

- The model is not just trained — it is **served via an API**
- The API validates its inputs with Pydantic schemas
- The pipeline is containerised with Docker so it runs identically everywhere
- There is a health-check endpoint, error handling, and logging

---

## Project Structure

```
01_sentiment_api/
├── README.md
├── notebooks/
│   └── 01_train_sentiment_model.ipynb  # Train and serialise the model
├── app/
│   ├── main.py                          # FastAPI application
│   ├── model.py                         # Model loading + prediction
│   └── schemas.py                       # Pydantic request/response schemas
├── models/
│   └── sentiment_pipeline.joblib        # Serialised TF-IDF + classifier
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Learning Objectives

- [ ] Train a TF-IDF + Logistic Regression pipeline and serialise it
- [ ] Build a FastAPI app with a `/predict` POST endpoint
- [ ] Validate request/response shapes with Pydantic
- [ ] Write a `Dockerfile` and run the app with `docker-compose up`
- [ ] Test the API from the command line with `curl`
- [ ] Add a `/health` endpoint for monitoring

---

## Key Concepts

| Concept | Why it matters |
|---------|---------------|
| sklearn `Pipeline` | Ensures TF-IDF and classifier are applied consistently at inference time |
| Pydantic validation | Prevents malformed requests from reaching the model |
| FastAPI async | Non-blocking request handling for concurrent predictions |
| Docker | "Works on my machine" becomes "works everywhere" |
| Health endpoint | Production services need a way to signal they are alive to load balancers |

---

## API Contract

```
POST /predict
Content-Type: application/json

{
  "text": "This product is absolutely terrible, broke after one day"
}

→ 200 OK
{
  "label": "negative",
  "confidence": 0.94,
  "processing_time_ms": 12
}
```

```
GET /health
→ 200 OK
{"status": "ok", "model_loaded": true}
```

---

## How to Run (once built)

```bash
# Option A: run directly
pip install -r requirements.txt
uvicorn app.main:app --reload

# Option B: Docker
docker-compose up --build

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product, very happy with my purchase!"}'
```
