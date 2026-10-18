
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Define model parameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

# 4. Enable automatic MLflow logging
mlflow.set_experiment("Iris-Classification")
mlflow.sklearn.autolog()

# 5. Train and evaluate the model
with mlflow.start_run():
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    # 6. Calculate test metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # 7. Log test metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("Test F1 Score:", f1)
