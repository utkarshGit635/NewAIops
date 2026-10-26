# Import TestClient so we can send simulated HTTP requests directly to the FastAPI app.
from fastapi.testclient import TestClient

# Import the application object defined in the main.py file.
from main import app

# Create a reusable test client for all test functions.
client = TestClient(app)

def test_health_endppoint():
    """
    Verify that the application is running and can respond to a health check request.
    
    """
    
    response = client.get("/health")
    
    # A successful http request should return a status code of 200.
    assert response.status_code == 200
    
    # Now check response body
    assert response.json() == {"status": "healthy"}


def test_valid_iris_prediction():
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
    
def test_invalid_iris_prediction():
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