from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# --------------------------------------------------
# Safe Absolute Model Path
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert_model")

# --------------------------------------------------

app = FastAPI(title="Disaster Text Classification API")

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # important for inference

# Label mapping
label_map = {
    0: "Not Disaster",
    1: "Disaster"
}

# Request body schema
class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Model is running successfully 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: TextRequest):
    with torch.no_grad():  # disables gradient calculation
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    return {
        "label": label_map[prediction],
        "confidence": round(confidence, 4)
    }