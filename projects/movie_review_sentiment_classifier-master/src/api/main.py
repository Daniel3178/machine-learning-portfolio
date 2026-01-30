from fastapi import FastAPI
from ml.model import MLModel
from logging_utils.logger import get_logger
from api.routes import predict

logger = get_logger("api")

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment prediction using a fine-tuned transformer model",
    version="0.0.1"
)

@app.on_event("startup")
def startup_event():
    """
    Load the model when the FastAPI app starts.
    """
    model_dir = "../../models/model_20251024_230330"
    num_labels = 2

    logger.info("Starting FastAPI and loading model...")
    try:
        model = MLModel.load_model(
            load_dir=model_dir,
            num_labels=num_labels,
            recreate_trainer=False
        )
        app.state.review_classifier = model
        logger.info(f"Model successfully loaded from {model_dir}")
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        app.state.review_classifier = None


@app.get("/")
def root():
    logger.info("Health check called.")
    return {"message": "Welcome to the Sentiment Analysis API!"}


app.include_router(predict.router, prefix="/api", tags=["Prediction"])

@app.on_event("shutdown")
def shutdown_event():
    logger.info("FastAPI is shutting down.")
