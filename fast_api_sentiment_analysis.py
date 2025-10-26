import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disables GPU entirely
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


app = FastAPI()

model_path = r"D:\sentiment_analysis\sentiment_analysis_amazon_reviews\sentiment_model"  # relative path from your FastAPI app folder

os.makedirs(model_path, exist_ok=True)


model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextIn):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits).item()
    return {"prediction": pred}
