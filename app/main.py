from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("app/models/mnist_model.pkl")

class MNISTInput(BaseModel):
    data: list

@app.post("/predict/")
async def predict(input_data: MNISTInput):
    try:
        data = np.array(input_data.data).reshape(1, -1)
        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))