#!/usr/bin/env python
"""
Random Search for Hyper-Parameter Optimization
------------------------
Part 1: 
Decision Trees (DT)
------------------------
Task 1: Hyperparameter Tuning (Random Search)
========================
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a DT classiWier.
4. Perform a Random Search to find the best hyperparameters for the DT classiWier. Search for
hyperparameters like max depth, min samples split, min samples leaf, and criterion. HInt: Use the
RandomizedSearchCV function from scikit-learn.
5. Print the best hyperparameters and the model’s accuracy with these hyperparameters.

Task 2: Error Analysis
========================
1. After training the DT model with the best hyperparameters from Task 1, use this model to make
predictions on the test data.
2. Identify and print the indices of misclassiWied instances (where the true class is not equal to the
predicted class).

Task 3: Confusion Matrix
========================
1. Calculate the confusion matrix for the model’s predictions on the test data.
2. Print the confusion matrix values (True Positives, True Negatives, False Positives, False
Negatives).
Note: The following Tasks 4 and 5 were not taught extensively in class for DTs. However, the
concepts were covered in liner regression, so I’d like you to give these a try w.r.t. DTs.

Task 4: Regression with DTs
========================
1. Load a dataset suitable for regression (e.g., the Boston housing dataset from scikit-learn).
2. Split the dataset into training and testing sets.
3. Implement a DT regression model.
4. Train the model on the training data.
5. Calculate and print the mean squared error (MSE) on the testing data to assess the model’s
performance.

Task 5: Metrics Comparison
========================
1. Compare the performance of the DT classiWier from Task 1 and the DT regression model from
Task 4.
2. Calculate and print relevant evaluation metrics for the classiWier (e.g., accuracy, precision, recall,
F1-score) and the regression model (e.g., MSE).
3. Discuss the results, including which model performed better and why.
"""
import numpy as np


