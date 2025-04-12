from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import predict  # Ensure this path is correct

app = FastAPI(title="Insomnia Prediction API")

class InputData(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int

@app.post("/predict")
def get_prediction(data: InputData):
    result = predict(data.dict())
    return {"prediction": result}
