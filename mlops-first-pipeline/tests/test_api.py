# Import TestClient so we can send simulated HTTP requests directly to the FastAPI app.
from fastapi.testclient import TestClient
from main import create_app
import pytest

# The test will no longer require a running MLFlow server or a real model, as we will use a fake model loader and version loader for testing purposes.
class FakeIrisModel:
    def predict(self, features):
        return [0]  # Always predict the first class (setosa) for testing purposes
    
def fake_model_loader():
    return FakeIrisModel()

def fake_version_loader():
    return "test_version"

@pytest.fixture
def client():
    # Create a FastAPI app instance with the fake model loader and version loader
    app = create_app(model_loader=fake_model_loader, version_loader=fake_version_loader)
    
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """
    Verify that the application is running and can respond to a health check request.
    
    """
    
    response = client.get("/health")
    assert response.status_code == 200
    
    result = response.json()
    assert result["status"] == "healthy"
    assert result["model_name"] == "IrisRandomForestModel"
    assert result["model_alias"] == "champion"
    assert result["model_version"] == "test_version"


def test_valid_iris_prediction(client):
    """
    Verify that valid flower measurements return a valid prediction.

    """

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)

    # A successful http request should return a status code of 200.
    assert response.status_code == 200

    # Convert the JSON response to a Python dictionary for easier assertions
    result = response.json()

    # Check that the response contains the expected keys
    assert "prediction" in result
    assert "class_name" in result

    # Check that the class_name is one of the expected iris species
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]

    # Check that the prediction is one of the expected class indices (0, 1, or 2)
    assert result["prediction"] in [0, 1, 2]
    
def test_invalid_iris_prediction(client):
    """
    Verify that invalid flower measurements are rejected by the API and return a 422 Unprocessable Entity error.

    """

    payload = {
        "sepal_length": -5.1,  # Invalid negative value
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # Send a simulated HTTP POST request with JSON input
    response = client.post("/predict", json=payload)

    # An invalid input should return a status code of 422.
    assert response.status_code == 422

    # Convert the JSON response to a Python dictionary for easier assertions
    result = response.json()

    # Check that the response contains the expected keys for validation errors
    assert "detail" in result