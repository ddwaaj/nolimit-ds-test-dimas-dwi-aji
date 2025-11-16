from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

from sentence_transformers import SentenceTransformer

# Load Artifacts
stop_words = joblib.load("../Program/Artifacts/stopwords.pkl")
embedder = SentenceTransformer("../Program/embedder_model")
clf = joblib.load("../Program/Artifacts/sentiment_classifier.pkl")

app = FastAPI(
    title="Gojek Sentiment Analysis API",
    description="API untuk prediksi sentiment komentar aplikasi Gojek",
    version="1.0.0"
)

# Cleaning Function
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Prediction Function
def predict_sentiment(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    pred = clf.predict(embedding)[0]
    proba = clf.predict_proba(embedding)[0].tolist()

    return {
        "input": text,
        "cleaned": cleaned,
        "prediction": pred,
        "probabilities": {
            "negatif": proba[0],
            "netral": proba[1],
            "positif": proba[2],
        }
    }

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict(request: TextInput):
    return predict_sentiment(request.text)