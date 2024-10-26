#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Task 1: Hyperparameter Tuning (Random Search)

# 1. Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# 4. Define the parameter distribution for Random Search
param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'criterion': ['gini', 'entropy']
}

# Perform Random Search
random_search = RandomizedSearchCV(
    dt,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    random_state=42,
    n_jobs=-1
)

# Fit the random search
random_search.fit(X_train, y_train)

# 5. Print the best hyperparameters and accuracy
print("Best Hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

print(f"\nBest Cross-Validation Score: {random_search.best_score_:.4f}")

# Get the best model
best_dt = random_search.best_estimator_

# Calculate accuracy on test set
test_accuracy = best_dt.score(X_test, y_test)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Task 2: Error Analysis

# 1. Make predictions on test data
y_pred = best_dt.predict(X_test)

# 2. Identify misclassified instances
misclassified_indices = np.where(y_test != y_pred)[0]

# Create a DataFrame to show misclassified instances with details
misclassified_data = pd.DataFrame({
    'Index in Test Set': misclassified_indices,
    'True Class': [iris.target_names[y_test[i]] for i in misclassified_indices],
    'Predicted Class': [iris.target_names[y_pred[i]] for i in misclassified_indices],
    'Feature Values': [X_test[i] for i in misclassified_indices]
})

print("\nMisclassified Instances:")
print(misclassified_data.to_string())

# Additional Analysis: Feature importance
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': best_dt.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))


