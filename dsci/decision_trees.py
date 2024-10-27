#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Set random seed
np.random.seed(73)
# Number of iterations
NUM_ITERATIONS = 100

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
# Larger values risk overfitting.
# "min_sample_split" -> Number of sample to split node w/ higher values meaning 
params = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, None],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "criterion": ["gini", "entropy"]
}

random_search = RandomizedSearchCV(
    dt,
    param_distributions=params,
    n_iter=NUM_ITERATIONS,
    cv=5,
    random_state=73,
    n_jobs=-1
)

# Fitting
random_search.fit(X_train, y_train)

# Output best hyperparameters and accuracy
print(f"\nBest Cross-validation Score: {random_search.best_score_:.4f}")
#
# Assess accuracy on best model via test set
best_dt = random_search.best_estimator_
test_accuracy =  best_dt.score(X_test, y_test)
print(f"Test set accuracy: {test_accuracy:.4f}")

y_predict = best_dt.predict(X_test)

# Find missclassifications
misclass_indices = np.where(y_test != y_predict)[0]

# Make dataframe to show the missclassifications
misclass_indices = pd.DataFrame({
    'Index in Test Set': misclass_indices,
    'True Class': [iris.target_names[y_test[i]] for i in misclass_indices],
    'Predicted Class': [iris.target_names[y_predict[i]] for i in misclass_indices],
    'Feature Values': [X_test[i] for i in misclass_indices]
})



breakpoint()
