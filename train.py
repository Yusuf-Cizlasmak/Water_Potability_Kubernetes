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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import seaborn as sns
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

def balance_data_with_adasyn(X, y):
    """Balance the dataset using ADASYN and return the balanced dataset with labels."""
    logging.info("Balancing dataset with ADASYN...")

    # Eksik değerleri doldur
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    adasyn = ADASYN(sampling_strategy='minority', random_state=RANDOM_STATE)
    X_res, y_res = adasyn.fit_resample(X_imputed, y)
    
    return X_res, y_res, adasyn



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

def build_pipeline() -> Pipeline:
    """Build a machine learning pipeline."""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),  # Handle missing values
        ('scaler', StandardScaler()),  # Feature scaling
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

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Balance the training data with ADASYN
    X_train_balanced, y_train_balanced, adasyn = balance_data_with_adasyn(X_train, y_train)

    # Build and train the pipeline
    model_pipeline = build_pipeline()

    # Cross-Validation
    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(model_pipeline, X_train_balanced, y_train_balanced, cv=CV_FOLDS)
    logging.info(f"Cross-validation scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    model_pipeline.fit(X_train_balanced, y_train_balanced)

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
