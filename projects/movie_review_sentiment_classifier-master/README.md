# Movie Review Sentiment Classifier

A machine learning project that classifies movie reviews as positive or negative using a fine-tuned transformer model. This project includes both training notebooks and a FastAPI service for making predictions.

## Overview

This project leverages transformer models (e.g., BERT) to perform sentiment analysis on movie reviews. It includes data preprocessing, model training, evaluation, and a production-ready API endpoint for sentiment predictions.

## Features

- **Fine-tuned Transformer Models**: Uses Hugging Face transformers for state-of-the-art sentiment classification
- **Data Preprocessing**: Complete pipeline for handling and preprocessing review text
- **Model Training**: Training framework using PyTorch and Hugging Face Trainer
- **REST API**: FastAPI-based service for real-time sentiment predictions
- **Logging**: Comprehensive logging throughout the application
- **Jupyter Notebooks**: Exploration and analysis notebooks for experimentation

## Project Structure

```
movie_review_sentiment_classifier/
├── src/
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # FastAPI app setup and endpoints
│   │   ├── routes/
│   │   │   └── predict.py       # Prediction endpoints
│   │   └── logs/                # API logs
│   ├── ml/                       # Machine learning module
│   │   ├── __init__.py
│   │   ├── model.py             # Model training and inference
│   │   ├── data.py              # Data loading and preprocessing
│   │   └── utils.py             # Utility functions
│   └── logging_utils/            # Logging configuration
│       ├── __init__.py
│       └── logger.py            # Logger setup
├── notebooks/                    # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   └── movie_review_sentiment_classifier.ipynb
├── data/                         # Data directory (raw and processed)
│   ├── raw/                     # Raw dataset files
│   └── processed/               # Processed dataset files
├── models/                       # Trained models
│   └── model_*/                 # Model artifacts and checkpoints
├── setup.py                      # Package setup configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
├── NOTE.MD                       # Project notes and deployment info
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Jupyter

See `requirements.txt` for the complete list of dependencies.

## Installation

### 1. Create a Python Virtual Environment

```bash
python -m venv .myenv
```

### 2. Activate the Virtual Environment

**On Windows:**
```bash
.myenv\Scripts\activate
```

**On macOS/Linux:**
```bash
source .myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Package in Development Mode

```bash
pip install -e .
```

### 5. (Optional) Create a Jupyter Kernel

```bash
python -m ipykernel install --user --name=myenv --display-name "my_ml_kernel"
```

## Usage

### Training the Model

Use the Jupyter notebooks to train the model:

1. Open `notebooks/movie_review_sentiment_classifier.ipynb`
2. Follow the cells to preprocess data and train the model
3. Models are saved to the `models/` directory

### Running the API

Start the FastAPI server:

```bash
cd src/api
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Making Predictions

**Health Check:**
```bash
curl http://localhost:8000/
```

**Predict Sentiment:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely amazing!"}'
```

## Development

### Project Setup Steps

1. Create virtual environment
2. Activate the environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install package in dev mode: `pip install -e .`
5. Create Jupyter kernel (optional): `python -m ipykernel install --user --name=myenv --display-name "my_ml_kernel"`

### Useful Commands

```bash
# List installed packages
pip list

# Uninstall the package
pip uninstall ml_project_classifier

# Run Jupyter notebook
jupyter notebook
```

## Model Information

The project uses fine-tuned transformer models for sequence classification:

- **Model Architecture**: AutoModelForSequenceClassification from Hugging Face
- **Tokenizer**: AutoTokenizer for text encoding
- **Training Framework**: PyTorch + Hugging Face Trainer
- **Metrics**: F1 Score, Accuracy

## Logging

The project includes comprehensive logging through the `logging_utils` module. Logs are written to both console and file outputs, helping track:

- Model initialization
- Training progress
- API requests
- Predictions and errors

## Deployment

For production deployment, refer to `NOTE.MD` for detailed deployment notes including:

- Docker containerization
- API integration
- Model serving strategies

## Author

Daniel Ibrahimi

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Troubleshooting

### Virtual Environment Issues
- Ensure you're in the correct virtual environment before running commands
- Use `pip list` to verify installed packages

### Model Loading Errors
- Check that the model directory path is correct in `src/api/main.py`
- Ensure all required model files are present in the models directory

### CUDA/GPU Issues
- The project automatically configures device (GPU/CPU) in `ml/utils.py`
- Check CUDA availability with `torch.cuda.is_available()`
