# Disaster Text Classification API

This project is a FastAPI-based NLP classification service built using DistilBERT.

## Model
- distilbert-base-uncased
- Fine-tuned on Kaggle Disaster Tweets dataset
- F1 Score: ~0.81

## Features
- REST API with FastAPI
- Predict disaster vs non-disaster text
- Confidence score output
- Health check endpoint

## Run Locally

1. Install dependencies:
pip install -r requirements.txt

2. Start server:
uvicorn app:app --reload

3. Open browser:
http://127.0.0.1:8000/docs

## API Endpoints

GET /
GET /health
POST /predict

Example request:

{
  "text": "There is a massive earthquake in California"
}

Example response:

{
  "label": "Disaster",
  "confidence": 0.92
}