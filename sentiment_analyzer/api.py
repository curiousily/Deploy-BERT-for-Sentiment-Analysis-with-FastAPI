from typing import List

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .model.sentiment_classifier import Model, get_model

app = FastAPI()


class PredictRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):

    probabilities: List[float]
    sentiment: str
    confidence: float


@app.post("/predict", response_model=SentimentResponse)
def predict(request: PredictRequest, model: Model = Depends(get_model)):
    sentiment, confidence, probabilities = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )
