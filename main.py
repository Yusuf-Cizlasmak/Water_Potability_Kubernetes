from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

# Read models saved during the training phase

model_loaded = joblib.load("saved_models/CatBoostModel.pkl")

# Here we are defining the types of our data.
class Water_Potability(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

    class Config:
        schema_extra = {
            "example": {
                "ph": 7.0,
                "Hardness": 150.4,
                "Solids": 48.0,
                "Chloramines": 7.16,
                "Sulfate": 183.0,
                "Conductivity": 500.0,
                "Organic_carbon": 13.28,
                "Trihalomethanes": 66.68,
                "Turbidity": 4.435
            }
        }

# Now let's set up FASTAPI...

app = FastAPI()

def make_prediction(model, request):
    # Parsing incoming data
    ph = request["ph"]
    Hardness = request["Hardness"]
    Solids = request["Solids"]
    Chloramines = request["Chloramines"]
    Sulfate = request["Sulfate"]
    Conductivity = request["Conductivity"]
    Organic_carbon = request["Organic_carbon"]
    Trihalomethanes = request["Trihalomethanes"]
    Turbidity = request["Turbidity"]

    # Creating the input vector
    scenario = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]

    # Prediction
    prediction = model.predict(scenario)

    return int(prediction[0])

# Set up the prediction FastAPI endpoint
@app.post("/prediction/water_potability")
def predict(request: Water_Potability):
    prediction = make_prediction(model_loaded, request.dict())
    return {"prediction": prediction}
