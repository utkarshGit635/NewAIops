import logging
import os
from pathlib import Path

import joblib
from fastapi import FastAPI
import mlflow
from mlflow.exceptions import MlflowException
from mlflow import MlflowClient
import mlflow.sklearn
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
)
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IrisRandomForestModel")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
LOCAL_MODEL_PATH = Path(__file__).parent / "models" / "iris_model.pkl"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Iris Classification")


def load_model():
    """Load the configured MLflow model, falling back to the local artifact."""
    try:
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS,
        ).version
        loaded_model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("Loaded model %s version %s", MODEL_URI, model_version)
        return loaded_model, str(model_version)
    except MlflowException as mlflow_error:
        if not LOCAL_MODEL_PATH.exists():
            raise RuntimeError(
                f"Could not load {MODEL_URI} from MLflow and local model "
                f"does not exist at {LOCAL_MODEL_PATH}: {mlflow_error}"
            ) from mlflow_error

        logger.warning(
            "Could not load %s from MLflow; using local model %s: %s",
            MODEL_URI,
            LOCAL_MODEL_PATH,
            mlflow_error,
        )
        return joblib.load(LOCAL_MODEL_PATH), "unknown"


model, model_version = load_model()


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classification API!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": model_version,
    }


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