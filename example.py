import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset with full information
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create DataFrame for better visualization
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y
df['target_names'] = pd.Categorical.from_codes(y, iris.target_names)

# Print dataset information
print("\n=== Dataset Information ===")
print(f"Dataset Shape: {X.shape}")
print(f"Number of Classes: {len(iris.target_names)}")
print(f"Target Classes: {iris.target_names.tolist()}")

print("\nFeature Names:")
for name in iris.feature_names:
    print(f"- {name}")

print("\nData Summary:")
print(df.describe())

print("\nClass Distribution:")
print(df['target_names'].value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== Train-Test Split Information ===")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Set up MLflow experiment
mlflow.set_experiment("iris_logistic_regression_experiment")

# Define the model with corrected parameters
# Removed multi_class parameter to avoid the warning
params = {
    "solver": "lbfgs",
    "max_iter": 400,
    "random_state": 8888
}

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)
    
    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = lr.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    
    # Log the model
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(lr, "model", signature=signature)
    
    # Print metrics
    print("\n=== Model Performance Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
