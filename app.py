from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import numpy as np
import joblib
import os

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI()

# ---------------------------------------------------
# Enable CORS
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Templates
# ---------------------------------------------------

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")

scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(model_path)

scaler = joblib.load(scaler_path)

# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------

class HouseData(BaseModel):

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# ---------------------------------------------------
# Home Route
# ---------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------

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

    return {
        "Predicted Price": float(prediction[0])
    }