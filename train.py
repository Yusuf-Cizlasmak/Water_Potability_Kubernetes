import os
import pathlib
import logging
from typing import Tuple, List

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable parameters
DATA_PATH = "water_potability.csv"
MODEL_DIR = "saved_models"
MODEL_NAME = "KNN.pkl"
TEST_SIZE = 0.07
RANDOM_STATE = 24
CV_FOLDS = 5

KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,
}

def vis_report_classification(y_true: List[int], y_pred: List[int]) -> pd.DataFrame:
    """Generate a classification report as a DataFrame."""
    df = pd.DataFrame(
        classification_report(y_true, y_pred, digits=2, output_dict=True)
    ).T
    df["support"] = df["support"].apply(int)
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    try:
        logging.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target variable, handle missing values."""
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def build_pipeline():
    """Build a machine learning pipeline."""

    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', PowerTransformer()),
        ('balancer', ADASYN(random_state=RANDOM_STATE)),
        ('classifier', KNeighborsClassifier(**KNN_PARAMS))
    ])
    return pipeline

def save_model(model: Pipeline, save_path: str):
    """Save the trained model to a file."""
    joblib.dump(model, save_path)
    logging.info(f"Model saved to {save_path}")

def load_model(model_path: str) -> Pipeline:
    """Load a model from a file."""
    try:
        logging.info(f"Loading model from {model_path}")
        return joblib.load(model_path)
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        raise


def main():
    # Paths
    current_dir = pathlib.Path(__file__).parent.resolve()
    model_dir = current_dir / MODEL_DIR
    model_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    model_path = model_dir / MODEL_NAME

    # Load and preprocess data
    data = load_data(DATA_PATH)
    X, y = preprocess_data(data)

    # Split the data using StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    plt.figure(figsize=(10, 15))

    plt.subplot(1, 2, 1)
    plt.pie(y_train.value_counts(), labels=y_train.unique(), autopct='%1.2f%%', shadow=True)
    plt.title('Training Dataset')

    plt.subplot(1, 2, 2)
    plt.pie(y_test.value_counts(), labels=y_test.unique(), autopct='%1.2f%%', shadow=True)
    plt.title('Test Dataset')

    plt.tight_layout()
    plt.show()


    # Balance the training data with ADASYN

    # Build and train the pipeline
    model_pipeline = build_pipeline()




    # Cross-Validation
    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=CV_FOLDS)
    logging.info(f"Cross-validation scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    model_pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model_pipeline.predict(X_test)
    report = vis_report_classification(y_test, y_pred)
    logging.info(f"\n{report}")

    # Save the trained model
    save_model(model_pipeline, model_path)


    # Load the model and make a prediction
    model_loaded = load_model(model_path)

    # Sample input for manual testing
    X_manual_test = [[7.0, 150, 47580.991603, 7.166639, 183, 500, 13.894419, 66.687695, 4.435821]]
    prediction = model_loaded.predict(X_manual_test)
    logging.info(f"Prediction for manual test input: {prediction}")

if __name__ == "__main__":
    main()
