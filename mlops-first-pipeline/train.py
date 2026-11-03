from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
# Client API lets us manage registered models and model versions and their aliases
from mlflow import MlflowClient

# Configure MLFlow Tracking 
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Group related training runs under a common experiment
mlflow.set_experiment("Iris_Classification")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


n_estimators = 17
random_state = 36

model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.2
}

# Start a tracked training run
with mlflow.start_run() as run:
    # Record the training configuration before model fitting
    mlflow.log_params(model_parameters)
    
    # Build classifier using selected hyperparameters

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


    # Train the classifier
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Calculate the precision
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Precision: {precision:.2f}")
    
    # Calculate the recall
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Recall: {recall:.2f}")
    
    # Model Quality Gate
    # Define the minimum acceptable test accuracy for now
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected due to low accuracy: {accuracy:.2f}. Minimum required accuracy is {MIN_TEST_ACCURACY:.2f}."
        )
        
    print("Model passed the quality gate with acceptable accuracy.")

    # Log the metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    # Save the trained model
    model_path = Path("models") / "iris_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")
    
    # Model registration with MLflow
    mlflow.sklearn.log_model(
        sk_model=rf,
        name="iris_model",
        registered_model_name="IrisRandomForestModel",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )
    
    # Assign the champion alias to the newly registered model version
    client = MlflowClient()
    # Identify the current run ID to retrieve the model version
    # We use the run ID to get the model version associated with this run
    current_run_id = mlflow.active_run().info.run_id
    
    # Retrieve all versions registered under this model name
    registered_versions = client.search_model_versions(f"name='IrisRandomForestModel'")
    
    # Find the version created by the current run
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )
    if current_model_version is None:
        raise RuntimeError(
            f"No registered model version found for run {current_run_id}."
        )
    
    client.set_registered_model_alias(
        name="IrisRandomForestModel",
        alias="champion",
        version=current_model_version.version
    )

    print("Registered model : IrisRandomForestModel")