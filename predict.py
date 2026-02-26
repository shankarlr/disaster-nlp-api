import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

MODEL_PATH = "models/distilbert_model"
TEST_PATH = "data/test.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

test_df = pd.read_csv(TEST_PATH)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

trainer = Trainer(model=model)

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

submission = pd.DataFrame({
    "id": test_df["id"],
    "target": preds
})

submission.to_csv("submission.csv", index=False)

print("Submission file generated: submission.csv")


