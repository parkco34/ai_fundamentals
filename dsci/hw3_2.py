#!/usr/bin/env python
"""
Part 2: Support Vector Machine (SVM) - Linear and w/ RBF Kernel.
5pts. each task
Please complete the following tasks to explore Linear Support Vector Machines (LSVM) and Support
Vector Machines (SVM) with an RBF kernel. You can use Python and libraries such as scikit-learn to
implement and demonstrate your work.
2
Task 1: Linear Support Vector Machine (LSVM)
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a Linear Support Vector Machine (SVM) classiWier using scikit-learn.
4. Train the LSVM model on the training data.
5. Evaluate the LSVM model’s performance on the test data and report accuracy.
Task 2: Support Vector Machine (SVM) with RBF Kernel
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a Support Vector Machine (SVM) classiWier with an RBF kernel using scikit-learn.
4. Train the SVM model with the RBF kernel on the training data.
5. Evaluate the SVM model’s performance on the test data and report accuracy.
Task 3: Hyperparameter Tuning for SVM with RBF Kernel
1. Perform hyperparameter tuning for the SVM with an RBF kernel. Search for optimal values of
hyperparameters such as C and γ using Random Search.
2. Report the best hyperparameters for the SVM with the RBF kernel.
3. Train a new SVM model with the best hyperparameters and evaluate its performance on the test
data.
Task 4: Metrics Comparison
1. Calculate and compare relevant evaluation metrics (e.g., accuracy, precision, recall, F1-score) for
the LSVM from Task 1 and the SVM with an RBF kernel from Task 2.
2. Not graded: Discuss the differences in performance and characteristics between these models.
"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def linear_svm(X_train, X_test, y_train, y_test):
    """
    Implements and evaluates Linear SVM classifier.
    --------------------------------------------------------
    INPUT:
        X_train: (np.ndarray) Training data
        X_test: (np.ndarray) Training Labels
        y_train: (np.ndarray) 

    OUTPUT:
    """
    # Initialize Linear SVM
    linear_svm = SVC(kernel="linear", random_state=RANDOM_STATE)

    # Train model
    linear_svm.fit(X_train, y_train)

    # Prediction
    y_prediction = linear_svm.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_prediction)
    print("\Linear SVM results: ")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")

    return linear_svm, accuracy, y_prediction

def rbf_svm(X_train, X_test, y_train, y_test):
    """
    SVM classifier with RBF kernel (Radial Basis Function Kernel)
    -----------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    pass

