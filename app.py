from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for all origins for testing purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],   # Allows all HTTP methods
    allow_headers=["*"],   # Allows all HTTP headers
)

# Load your model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a data model
class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
async def predict(features: Features):
    input_data = np.array([[features.feature1, features.feature2, features.feature3, features.feature4]])
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}
