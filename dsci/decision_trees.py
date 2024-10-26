#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Set random seed
np.random.seed(73)

# Load dataset
iris = load_iris()
# X: Flower measurements -> sepal length, sepal width, petal length, petal width for
# 120 flowers
# y: Species of flower, 0, 1, or 2
X, y = iris.data, iris.target

# Split data
# Feature data used to train 80% since test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=73)

dt = DecisionTreeClassifier(random_state=73)
# Define parameter dist. for Random Search
# "max_depth": Determines how deep the tree goes, with None meaning no limit.
# The smaller the value, the simpler the tree, less likely to overfit
# Larger values risk overfitting
params = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, None],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "criteria": ["gini", "entropy"]
}



breakpoint()
