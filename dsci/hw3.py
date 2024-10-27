#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# initiai CONSTANTS
RANDOM_STATE = 73
# Feature data used to train 80% since test_size=0.2
TEST_SIZE = 0.2

# Set random seed
np.random.seed(RANDOM_STATE)

# TASK 1: =======================================================
# Load dataset
# X: Flower measurements -> sepal length, sepal width, petal length, petal width for
# 120 flowers
# y: Species of flower, 0, 1, or 2
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# Define parameter dist. for Random Search
# "max_depth": Determines how deep the tree goes, with None meaning no limit.
# The smaller the value, the simpler the tree, less likely to overfit
# Larger values risk overfitting.
# "min_sample_split" -> Number of sample to split node w/ higher values meaning 
# "min_sample_leaf" -> Samples required at each leaf node w/ higher values for
# better balanced trees
# Criterion -> Quality of split, where Gini measures probability of incorrect
# classification and Entropy measuress info gain
params = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, None],
    "min_sample_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_sample_leaf": [1,2, 3, 4, 5],
    "criterion": ["gini", "entropy"]
}




