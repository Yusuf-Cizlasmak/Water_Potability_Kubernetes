import os
import joblib
import pathlib
import pandas as pd
import logging
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def vis_report_classification(y_true: List[int], y_pred: List[int]) -> pd.DataFrame:
    """Generate a classification report as a DataFrame."""
    df = pd.DataFrame(
        classification_report(y_true, y_pred, digits=2, output_dict=True)
    ).T
    df["support"] = df["support"].apply(int)
    return df


def read_train():
    CV_FOLDS = 5
    data = pd.read_csv('water_potability.csv')

    # Features
    X = data.iloc[:, :-1]
    # Fill nan values
    # You can change the strategy as needed
    imputer = SimpleImputer(strategy='mean')

    # Fit the imputer to your data (usually, you would fit it to your training data)
    imputer.fit(X)  # X is your input data, a DataFrame or array

    # Transform the data by filling missing values
    X = imputer.transform(X)

    # Target value
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=72,shuffle=True)
    
    plt.figure(figsize=(10, 15))

    plt.subplot(1, 2, 1)
    plt.pie(y_train.value_counts(), labels=y_train.unique(), autopct='%1.2f%%', shadow=True)
    plt.title('Training Dataset')

    plt.subplot(1, 2, 2)
    plt.pie(y_test.value_counts(), labels=y_test.unique(), autopct='%1.2f%%', shadow=True)
    plt.title('Test Dataset')

    plt.tight_layout()
    plt.show()


    # Training
    KNN_PARAMS = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
    }

    model = KNeighborsClassifier(**KNN_PARAMS)

    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    # Cross-Validation


    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS)
    logging.info(f"Cross-validation scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    report = vis_report_classification(y_test, y_pred)
    logging.info(f"\n{report}")

    # Save Model

    current_dir = pathlib.Path(__file__).parent.resolve()
    print(f"current_dir: {current_dir}")
    directoryName = os.path.join(current_dir, 'saved_models')
    print(directoryName)

    joblib.dump(model, os.path.join(directoryName, "CatBoostModel.pkl"))

    # Make a predict
    model_loaded = joblib.load(os.path.join(
        directoryName, "CatBoostModel.pkl"))

    # ' ph    Hardness        Solids  Chloramines     Sulfate  Conductivity  Organic_carbon  Trihalomethanes  Turbidity
    X_manuel_test = [[7.0, 150, 47580.991603, 7.166639, 183,
                      500,    13.894419,     66.687695,  4.435821]]

    prediction = model_loaded.predict(X_manuel_test)
    print(f"prediction is {prediction}")


if __name__ == "__main__":
    read_train()