from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models") / "iris_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Classification")

class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1, gt= 0)
    sepal_width: float = Field(..., example=3.5, gt= 0)
    petal_length: float = Field(..., example=1.4, gt= 0)
    petal_width: float = Field(..., example=0.2, gt= 0)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classification API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    prediction = int(model.predict(features)[0])
    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }