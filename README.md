# Machine Learning Portfolio

A comprehensive machine learning portfolio containing foundational exercises, practical projects, and utilities for learning and implementing ML algorithms and models.

## 📋 Overview

This repository is organized into three main sections:
- **Exercises**: Foundational ML concepts with interactive Jupyter notebooks
- **Projects**: Real-world ML applications and case studies
- **Common**: Shared utilities and helper functions



## 📚 Exercises - Foundational ML

The `exercises/foundation/` directory contains 9 progressively complex Jupyter notebooks covering fundamental machine learning concepts:

| # | Notebook | Topics |
|---|----------|--------|
| 1 | [1.0_basic.ipynb](exercises/foundation/1.0_basic.ipynb) | ML basics and fundamentals |
| 2 | [2.0_linear_regression_&_svm.ipynb](exercises/foundation/2.0_linear_regression_&_svm.ipynb) | Linear Regression, Support Vector Machines |
| 3 | [3.0_generalization_&_evaluation_cm.ipynb](exercises/foundation/3.0_generalization_&_evaluation_cm.ipynb) | Generalization, Evaluation Metrics, Confusion Matrix |
| 4 | [4.0_neural_network_basic.ipynb](exercises/foundation/4.0_neural_network_basic.ipynb) | Neural Network Fundamentals |
| 5 | [5.0_dimensionality_reduction_PCA.ipynb](exercises/foundation/5.0_dimensionality_reduction_PCA.ipynb) | Dimensionality Reduction, PCA |
| 6 | [6.0_kernel_methods.ipynb](exercises/foundation/6.0_kernel_methods.ipynb) | Kernel Methods |
| 7 | [7.0_mixture_model_&_probability.ipynb](exercises/foundation/7.0_mixture_model_&_probability.ipynb) | Mixture Models, Probability Theory |
| 8 | [8.0_information_theory_decision_tree.ipynb](exercises/foundation/8.0_information_theory_decision_tree.ipynb) | Information Theory, Decision Trees |
| 9 | [9.0_generative_model_basic.ipynb](exercises/foundation/9.0_generative_model_basic.ipynb) | Generative Models |

## 🚀 Projects

### Movie Review Sentiment Classifier

**Location**: [projects/movie_review_sentiment_classifier-master/](projects/movie_review_sentiment_classifier-master/)

A production-ready sentiment analysis project using fine-tuned transformer models.

**Key Files**:
- [README.md](projects/movie_review_sentiment_classifier-master/README.md) - Project documentation
- [setup.py](projects/movie_review_sentiment_classifier-master/setup.py) - Package configuration
- [src/api/main.py](projects/movie_review_sentiment_classifier-master/src/api/main.py) - FastAPI application
- [src/ml/model.py](projects/movie_review_sentiment_classifier-master/src/ml/model.py) - Model training
- [src/ml/data.py](projects/movie_review_sentiment_classifier-master/src/ml/data.py) - Data handling
- [notebooks/movie_review_sentiment_classifier.ipynb](projects/movie_review_sentiment_classifier-master/notebooks/movie_review_sentiment_classifier.ipynb) - Main notebook

**Features**:
- Fine-tuned transformer models for sentiment classification
- REST API with FastAPI
- Comprehensive logging
- Data preprocessing pipeline
- Model training and evaluation

See the [project README](projects/movie_review_sentiment_classifier-master/README.md) for detailed setup and usage instructions.

### 5. Running Projects
Each project has its own setup instructions. See the individual project README:
```bash
cd projects/movie_review_sentiment_classifier-master/
pip install -r requirements.txt
pip install -e .
```

## 🎯 Quick Links

- **Exercises**: [exercises/foundation/](exercises/foundation/)
- **Projects**: [projects/](projects/)
- **Sentiment Classifier Project**: [projects/movie_review_sentiment_classifier-master/](projects/movie_review_sentiment_classifier-master/)
  - [Project README](projects/movie_review_sentiment_classifier-master/README.md)
  - [API Main File](projects/movie_review_sentiment_classifier-master/src/api/main.py)
  - [Model Implementation](projects/movie_review_sentiment_classifier-master/src/ml/model.py)

## 📁 Repository Structure

```
MachineLearningPortfolio/
├── README.md                          # This file
├── .gitignore                         # Git ignore patterns for entire portfolio
├── requirements.txt                   # Python dependencies
├── common/                            # Shared utilities (currently empty)
├── exercises/                         # Learning materials & exercises
│   └── foundation/                    # Foundational ML concepts
│       ├── 1.0_basic.ipynb           # Basic concepts
│       ├── 2.0_linear_regression_&_svm.ipynb
│       ├── 3.0_generalization_&_evaluation_cm.ipynb
│       ├── 4.0_neural_network_basic.ipynb
│       ├── 5.0_dimensionality_reduction_PCA.ipynb
│       ├── 6.0_kernel_methods.ipynb
│       ├── 7.0_mixture_model_&_probability.ipynb
│       ├── 8.0_information_theory_decision_tree.ipynb
│       └── 9.0_generative_model_basic.ipynb
└── projects/                          # Real-world ML projects
    └── movie_review_sentiment_classifier-master/
        ├── README.md                  # Project-specific documentation
        ├── .gitignore                 # Project-specific git patterns
        ├── setup.py                   # Package setup configuration
        ├── NOTE.MD                    # Deployment notes
        ├── requirements.txt           # Project dependencies
        ├── src/                       # Source code
        │   ├── api/                   # FastAPI application
        │   │   ├── main.py           # FastAPI app setup
        │   │   ├── logs/             # API logs
        │   │   └── routes/
        │   │       └── predict.py    # Prediction endpoints
        │   ├── ml/                    # Machine learning module
        │   │   ├── __init__.py
        │   │   ├── model.py          # Model training & inference
        │   │   ├── data.py           # Data loading & preprocessing
        │   │   └── utils.py          # Utility functions
        │   └── logging_utils/         # Logging configuration
        │       ├── __init__.py
        │       └── logger.py         # Logger setup
        ├── notebooks/                 # Jupyter notebooks
        │   ├── 01_exploration.ipynb
        │   ├── movie_review_sentiment_classifier.ipynb
        │   └── logs/                 # Notebook logs
        └── data/                      # Data storage
            ├── raw/                   # Raw datasets
            └── processed/             # Processed datasets
```


## 📝 Project Notes

- Detailed deployment and implementation notes for the sentiment classifier are available in [projects/movie_review_sentiment_classifier-master/NOTE.MD](projects/movie_review_sentiment_classifier-master/NOTE.MD)
- Each project directory has its own `.gitignore` for project-specific patterns
- The root [.gitignore](.gitignore) covers common patterns for the entire portfolio


