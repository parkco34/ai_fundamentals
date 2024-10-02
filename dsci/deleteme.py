#!/usr/bin/env python
# Part 1: Gradient Descent Implementation without StandardScaler

import numpy as np

# 1. Generate Synthetic Data
np.random.seed(42)  # For reproducibility

# Generate 200 random values of X between -10 and 10
X = np.random.uniform(-10, 10, 200).reshape(-1, 1)  # Shape (200, 1)

# Generate the target variable y using the function y = 5x^4 + 2x^3 + 9x + 12
y = 5 * X**4 + 2 * X**3 + 9 * X + 12  # Shape (200, 1)

# Initialize the bias term Xbias to 1
X_bias = np.ones((X.shape[0], 1))  # Shape (200, 1)

# Build the design matrix X_design with columns: bias term, x, x^3, x^4
X_design = np.hstack((X_bias, X, X**3, X**4))  # Shape (200, 4)

# 2. Implement Gradient Descent

# Function to compute Root Mean Squared Error (RMSE)
def compute_error(y_actual, y_pred):
    """
    Calculate the Root Mean Squared Error between actual and predicted values.

    Parameters:
    y_actual (numpy.ndarray): Actual target values.
    y_pred (numpy.ndarray): Predicted target values.

    Returns:
    float: RMSE value.
    """
    error = np.sqrt(np.mean((y_pred - y_actual) ** 2))
    return error

# Function to perform Gradient Descent and return optimized weights
def gradient_descent(X, y, learning_rate, n_iterations):
    """
    Perform Gradient Descent to optimize weights.

    Parameters:
    X (numpy.ndarray): Design matrix with input features.
    y (numpy.ndarray): Target values.
    learning_rate (float): Learning rate for weight updates.
    n_iterations (int): Number of iterations.

    Returns:
    numpy.ndarray: Optimized weights.
    """
    m = len(y)
    theta = np.random.randn(X.shape[1], 1)  # Random initialization of weights
    for i in range(n_iterations):
        y_pred = X.dot(theta)
        gradients = (2 / m) * X.T.dot(y_pred - y)
        theta -= learning_rate * gradients
    return theta

# 3. Hyperparameters and Experimentation
learning_rates = [0.001, 0.01, 0.0005]
n_iterations = 1000  # Number of iterations

# Run the experiment with different learning rates
for lr in learning_rates:
    # Perform Gradient Descent to optimize weights
    theta_optimal = gradient_descent(X_design, y, learning_rate=lr, n_iterations=n_iterations)

    # Predict using the optimized weights
    y_pred = X_design.dot(theta_optimal)

    # Compute RMSE on the original data
    rmse = compute_error(y, y_pred)

    # Display the final RMSE for each learning rate
    print(f"Learning Rate: {lr}; RMSE: {rmse:.4f}")


