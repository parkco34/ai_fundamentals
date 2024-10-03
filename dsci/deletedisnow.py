#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
iris = load_iris()
indices = np.where((iris.target == 0) | (iris.target == 1))
X = iris.data[indices][:, :2]  # Use only the first two features
y = iris.target[indices]

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
print(f"Actual classes: {y_test}")

# Performance Metrics
TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

print("\nPerformance Metrics:")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

y_pred_proba = log_reg.predict_proba(X_test_scaled)
print("Prediction probabilities:")
print(y_pred_proba)

cv_scores = cross_val_score(log_reg, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

# Visualize decision boundary
plt.figure(figsize=(10,6))
x1 = X[:, 0]
x2 = X[:, 1]

plt.scatter(x1[y==0], x2[y==0], c='blue', label='Class 0')
plt.scatter(x1[y==1], x2[y==1], c='red', label='Class 1')

x1_min, x1_max = x1.min() - 1, x1.max() + 1
x2_min, x2_max = x2.min() - 1, x2.max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                     np.arange(x2_min, x2_max, 0.1))
Z = log_reg.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.legend()
plt.show()
