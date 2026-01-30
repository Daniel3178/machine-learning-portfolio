import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
from ml.utils import configure_device, get_save_dir
from logging_utils.logger import get_logger
from typing import Optional
import torch
import os

class IMDbDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MLModel:

    def __init__(self, model_name: str, num_labels: int, x_train: list, x_test: list, y_train: list, y_test: list):
        self.logger = get_logger("MLModel")
        self.trainer = None
        self.__is_trained = False
        self.dataset = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

        try:
            self.logger.info(f"Initializing model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            device = configure_device()
            self.logger.info(f"Using device: {device}")
            model.to(device)
            self.model = model
            self.tokenizer = tokenizer
            self.logger.info("Model and tokenizer successfully initialized.")
        except Exception as e:
            self.logger.exception(f"Error initializing model: {e}")


    @staticmethod
    def _default_training_args():
        return TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",
        )
        
    @staticmethod
    def _default_compute_metrics():
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="weighted")
            }
        return compute_metrics
    
    @classmethod
    def run_pipeline(cls, x_train: list, x_test: list, y_train: list, y_test: list, ml_model: Optional["MLModel"] = None, model_name="distilbert-base-uncased", num_labels=2):
        logger = get_logger("Pipeline")
        if ml_model is None:
            logger.info("No model provided ? running default training pipeline.")
            ml_model = cls(model_name=model_name, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, num_labels=num_labels)
            ml_model.initialize_training_model(
                truncation=True, padding=True, 
                training_args=cls._default_training_args()
            )
            ml_model.train_model()
            ml_model.evaluate_model()
        else:
            ml_model.train_model()
            ml_model.evaluate_model()
        return ml_model

    @classmethod
    def load_model(cls, load_dir: str, num_labels: int, x_train: list = None, x_test: list = None, y_train: list = None, y_test: list = None, recreate_trainer: bool = False):
        logger = get_logger("ModelLoader")
        try:
            logger.info(f"Loading model from {load_dir}")
            tokenizer = AutoTokenizer.from_pretrained(load_dir)
            model = AutoModelForSequenceClassification.from_pretrained(load_dir, num_labels=num_labels)
            device = configure_device()
            logger.info(f"Using device: {device}")
            model.to(device)
            instance = cls.__new__(cls)
            instance.model = model
            instance.tokenizer = tokenizer
            instance.trainer = None
            instance.__is_trained = True
            instance.logger = logger
            instance.dataset = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

            if recreate_trainer:
                training_args_file = os.path.join(load_dir, "training_args.json")
                if os.path.exists(training_args_file):
                    with open(training_args_file, "r") as f:
                        training_args = json.load(f)
                    training_args = TrainingArguments(**training_args)
                    logger.info("Training arguments reloaded from file.")
                else:
                    training_args = cls._default_training_args()
                    logger.warning("Training arguments file not found. Using defaults.")

                train_encodings = tokenizer(x_train, truncation=True, padding=True)
                test_encodings = tokenizer(x_test, truncation=True, padding=True)
                train_dataset = IMDbDataset(train_encodings, y_train)
                test_dataset = IMDbDataset(test_encodings, y_test)
                compute_metrics = cls._default_compute_metrics()
                instance.trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    compute_metrics=compute_metrics
                )
                logger.info("Trainer successfully recreated.")

            logger.info("Model loaded successfully.")
            return instance

        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            return None

    def train_model(self, training_args: TrainingArguments = None, truncation: bool = True, padding: bool = True, save_model: bool = True, save_dir: str = "./models"):
        if self.__is_trained:
            self.logger.info("Model is already trained.")
            return True
        if training_args is None:
            training_args = MLModel._default_training_args()
        try:
            if self.trainer is None:
                self.initialize_training_model(truncation, padding, training_args)
            self.logger.info("Starting training...")
            self.trainer.train()
            self.__is_trained = True
            self.logger.info("Training completed.")
            if save_model:
                saved_dir = self.save_model(base_dir=save_dir)
                self.logger.info(f"Model trained and saved to {saved_dir}")
            return True
        except Exception as e:
            self.logger.exception(f"Error during training: {e}")
            return False

    def evaluate_model(self):
        if self.trainer is None:
            self.logger.warning("Trainer not initialized. Please train the model first.")
            return
        try:
            if not self.__is_trained:
                self.logger.warning("Evaluating an untrained model.")
            results = self.trainer.evaluate()
            self.logger.info(f"Evaluation results: {results}")
            return results
        except Exception as e:
            self.logger.exception(f"Error during evaluation: {e}")
            return False
    
    def initialize_training_model(self, truncation: bool, padding: bool, training_args: TrainingArguments = None, compute_metrics=None):       
        if training_args is None:
            training_args = MLModel._default_training_args()

        train_encodings = self.tokenizer(self.dataset['x_train'], truncation=truncation, padding=padding)
        test_encodings = self.tokenizer(self.dataset['x_test'], truncation=truncation, padding=padding)
        train_dataset = IMDbDataset(train_encodings, self.dataset['y_train'])
        test_dataset = IMDbDataset(test_encodings, self.dataset['y_test'])
        
        if compute_metrics is None:
            compute_metrics = MLModel._default_compute_metrics()
            
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            )
        return self.trainer

    def save_model(self, base_dir="./models"):
        try:
            save_dir = get_save_dir(base_dir)
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            if self.trainer is not None:
                training_args_file = os.path.join(save_dir, "training_args.json")
                args_json = self.trainer.args.to_json_string()
                with open(training_args_file, "w") as f:
                    json.dump(args_json, f, indent=4)
                self.logger.info(f"TrainingArguments saved to {training_args_file}")
            self.logger.info(f"Model and tokenizer saved to {save_dir}")
            return save_dir
        except Exception as e:
            self.logger.exception(f"Error saving model: {e}")
            return False

    def predict(self, texts: str | list[str], truncation=True, padding=True):
        if not self.__is_trained:
            self.logger.warning("Using an untrained model for prediction.")
        if isinstance(texts, str):
            texts = [texts]
        encodings = self.tokenizer(texts, truncation=truncation, padding=padding, return_tensors="pt")
        device = configure_device()
        self.logger.info(f"Using device: {device}")
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
        self.logger.info(f"Predicted {len(predictions)} samples.")
        mapping = {0: "negative", 1: "positive"}
        return [mapping[p] for p in predictions.cpu().tolist()]
