#!/usr/bin/env python
"""
Implementing Gradient Descent to minimize cost function of synthetic dataset
generated from a non-linear function.
"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(73)

# Generate 200 randomly generated values of X between −10 and 10
X = np.random.uniform(-10, 10, 200).reshape(-1, 1)
m = len(X)
# Target value
y = 5 * X**4 + 2 * X**3 + 9 * X + 12

# Bias term; combining a column of ones w/ shape (m,1) and original X values
X_bias = np.c_[np.ones((m, 1)), X] # Fits data that might not pass thru origin

# Hyperparameters
learning_rates = [0.001, 0.01, 0.0005]
# Polynomial Regression; adding powers of each feature as new features
epochs = 1000

def compute_error(y_actual, y_predicted):
    """
    Computes the Root Mean Squared Error (RMSE) between actual/predicted y values.
    ---------------------------------------
    INPUT:
        y_actual: numpy.ndarray
        y_predicted: numpy.ndarray

    OUTPUT:
        numpy.float64
    """
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

def gradient_descent(X, y, learning_rate, epochs):
    """
    Perform Gradient Descent for weight optimization.
    --------------------------------------------------
    INPUT:
        X: numpy.ndarray
        y: numpy.ndarray
        learning_rate: float
        epochs: int
    OUTPUT:
        theta: numpy.ndarray
    """
    m, n = X.shape  # m is number of samples, n is number of features
    # initialize weights randomly
    theta = np.random.randn(n, 1)

    for epoch in range(epochs):
        y_pred = X.dot(theta)
        gradients = (2 / m) * X.T.dot(y_pred - y)
        # Update weights
        theta -= learning_rate * gradients
    return theta

# Experiments
for lr in learning_rates:
    # OPtimize weights
    optimal_theta = gradient_descent(X_bias, y, learning_rate=lr, epochs=epochs)
    # Predictions
    y_pred = X_bias.dot(optimal_theta)
    # Compute error
    rmse = compute_error(y, y_pred)
    print(f"Learning Rate: {lr:.4f}; RMSE: {rmse:.4f}")








