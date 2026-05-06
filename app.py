from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(data: HouseData):

    input_data = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    return {"Predicted Price": float(prediction[0])}