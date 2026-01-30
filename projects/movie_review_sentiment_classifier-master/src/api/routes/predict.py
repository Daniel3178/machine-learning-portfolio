from fastapi import APIRouter, Request
from pydantic import BaseModel
from logging_utils.logger import get_logger

router = APIRouter()
logger = get_logger("api")

class Review(BaseModel):
    text: str


@router.post("/predict")
def predict_sentiment(review: Review, request: Request):
    """
    Predict sentiment for a given text review.
    """
    model = getattr(request.app.state, "review_classifier", None)

    if model is None:
        logger.warning("Prediction attempted before model was loaded.")
        return {"error": "Model is not loaded yet. Try again later."}

    try:
        prediction = model.predict(review.text)
        # label = "positive" if prediction[0] == 1 else "negative"
        logger.info(f"Prediction made successfully: '{review.text[:50]}...' -> {prediction}")
        return {
            "sentiment": prediction,
            "prediction_raw": prediction
        }
    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        return {"error": str(e)} 
