from fastapi import FastAPI
from pydantic import BaseModel
import joblib

MODEL_PATH = "models/text_classifier.joblib"

# Load model once
model = joblib.load(MODEL_PATH)

app = FastAPI()

class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Restaurant Review Classifier API"}


@app.post("/predict")
def predict(req: ReviewRequest):
    text = req.text

    label = model.predict([text])[0]

    return {
        "text": text,
        "predicted_label": label
    }
