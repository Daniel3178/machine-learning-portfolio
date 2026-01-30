from sklearn.model_selection import train_test_split
import pandas as pd
import re

def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/><br\s*/>", " ", text)  # Replace HTML line breaks with space
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
    text = text.lower()
    return text

class DataPreprocessor:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def transform(self, target: str, transformer):
        self.dataset[target] = self.dataset[target].map(transformer)
        return self.dataset[target]

    def split_data(self, feature:str, target: str, test_size: float = 0.2, random_state: int = 13):
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset[feature], self.dataset[target], test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test