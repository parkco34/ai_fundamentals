#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
iris = load_iris()
indices = np.where((iris.target == 0) | (iris.target == 1))
X = iris.data[indices[0], :2]  # Use only the first two features (sepal length and sepal width)
y = iris.target[indices[0]]    # Target labels for classes 0 and 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Prediction
y_pred = log_reg.predict(X_test_scaled)
print(f"Predicted classes: {y_pred}")
print(f"Actual classes:    {y_test}")

# Performance Metrics Calculation (without using sklearn)
TP = np.sum((y_test == 1) & (y_pred == 1))  # True Positives
TN = np.sum((y_test == 0) & (y_pred == 0))  # True Negatives
FP = np.sum((y_test == 0) & (y_pred == 1))  # False Positives
FN = np.sum((y_test == 1) & (y_pred == 0))  # False Negatives

print("\nPerformance Metrics:")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Calculate RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")

# Visualization with decision boundary

# Combine the training and test data for visualization
X_combined_scaled = np.vstack((X_train_scaled, X_test_scaled))
y_combined = np.hstack((y_train, y_test))

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(X_combined_scaled[y_combined == 0][:, 0], X_combined_scaled[y_combined == 0][:, 1],
            color='blue', marker='o', label='Setosa')
plt.scatter(X_combined_scaled[y_combined == 1][:, 0], X_combined_scaled[y_combined == 1][:, 1],
            color='red', marker='s', label='Versicolor')

# Create a mesh to plot the decision boundary
x_min, x_max = X_combined_scaled[:, 0].min() - 1, X_combined_scaled[:, 0].max() + 1
y_min, y_max = X_combined_scaled[:, 1].min() - 1, X_combined_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Use the logistic regression model to predict the class for each point in the mesh
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='green')

plt.xlabel('Standardized Sepal Length')
plt.ylabel('Standardized Sepal Width')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()




