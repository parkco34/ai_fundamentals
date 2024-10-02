#!/usr/bin/env python
# ------------------ Part 2: Logistic Regression Implementation ------------------

# 1. Load and Preprocess the Iris Dataset

# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()

# Extract only the first two classes to make it a binary classification problem
# Target classes: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
# We will use classes 0 and 1
X = iris.data[iris.target != 2]  # Features for classes 0 and 1
y = iris.target[iris.target != 2]  # Target labels for classes 0 and 1

# Split the dataset into training and testing sets
# Use test_size=0.2 and random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model Training

# Train a Logistic Regression model on the training data with default parameters
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Make Predictions

# Predict the classes on the test set
y_pred = model.predict(X_test)
print("Predicted Classes:")
print(y_pred)

# 4. Performance Metrics Calculation (NO using sklearn)

# Manually calculate True Positives (TP), True Negatives (TN),
# False Positives (FP), and False Negatives (FN)

# Define positive class as 1 and negative class as 0
TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Manually calculate and print the Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


