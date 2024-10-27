#!/usr/bin/env python
"""
SVM
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

# Generate the dataset
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Create and train the polynomial SVM classifier
polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42)
)
polynomial_svm_clf.fit(X, y)

# Create a grid of points to plot the decision boundary
x0s = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
x1s = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
x0, x1 = np.meshgrid(x0s, x1s)
X_grid = np.c_[x0.ravel(), x1.ravel()]

# Predict the labels for each point in the grid
y_pred = polynomial_svm_clf.predict(X_grid).reshape(x0.shape)

# Plot the decision boundary and the data points
plt.figure(figsize=(10, 6))
plt.contourf(x0, x1, y_pred, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='coolwarm', edgecolors='k')
plt.xlabel('$x_0$', fontsize=14)
plt.ylabel('$x_1$', fontsize=14)
plt.title('Decision Boundary of Polynomial SVM Classifier', fontsize=16)
plt.show()

