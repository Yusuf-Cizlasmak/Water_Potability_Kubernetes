import os
import joblib
import pathlib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score


from catboost import CatBoostClassifier


def read_train():

    data = pd.read_csv('water_potability.csv')
    print(data)

    # Features
    X = data.iloc[:, :-1]
    print(X)
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
        X, y, test_size=0.07, random_state=24)

    # Training
    param = {

        'l2_leaf_reg': 0.0028772916926549765, 'max_bin': 390, 'learning_rate': 0.011398945601759285, 'n_estimators': 379, 'max_depth': 9, 'random_state': 2021
    }
    model = CatBoostClassifier(**param)

    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(f"R2: {r2}")

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



