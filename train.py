import os
import pathlib

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def vis_report_classification(y_hat_xgboost, y_test):
    df = pd.DataFrame(
        classification_report(y_hat_xgboost, y_test, digits=2, output_dict=True)
    ).T

    df["support"] = df.support.apply(int)

    return df


def read_train():
    data = pd.read_csv("water_potability.csv")
    # Features
    X = data.iloc[:, :-1]
    # Fill nan values
    # You can change the strategy as needed
    imputer = SimpleImputer(strategy="mean")

    # Fit the imputer to your data (usually, you would fit it to your training data)
    imputer.fit(X)  # X is your input data, a DataFrame or array

    # Transform the data by filling missing values
    X = imputer.transform(X)

    # Target value
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.07, random_state=24
    )

    # Training
    param = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
    }

    model = KNeighborsClassifier(**param)

    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print(vis_report_classification(y_pred, y_test))

    # Save Model

    current_dir = pathlib.Path(__file__).parent.resolve()
    print(f"current_dir: {current_dir}")
    directoryName = os.path.join(current_dir, "saved_models")
    print(directoryName)

    joblib.dump(model, os.path.join(directoryName, "KNN.pkl"))

    # Make a predict
    model_loaded = joblib.load(os.path.join(directoryName, "KNN.pkl"))

    # ' ph    Hardness        Solids  Chloramines     Sulfate  Conductivity  Organic_carbon  Trihalomethanes  Turbidity
    X_manuel_test = [
        [7.0, 150, 47580.991603, 7.166639, 183, 500, 13.894419, 66.687695, 4.435821]
    ]

    prediction = model_loaded.predict(X_manuel_test)
    print(f"prediction is {prediction}")


if __name__ == "__main__":
    read_train()
