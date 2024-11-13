#!/usr/bin/env python
"""
Part 1: 
Decision Trees (DT)
------------------------
Task 1: Hyperparameter Tuning (Random Search)
========================
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a DT classifier.
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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random import random, choice
import DecisionTree as dt

# Constants
TEST_SIZE = 0.3
N_ITERATIONS = 100
RANDOM_STATE = 73
np.random.seed(RANDOM_STATE)

def define_params(
    max_depth=[i for i in range(1, 21)], 
    min_samples_split=
        [i for i in range(2, 21)],
    min_samples_leaf=
        [i for i in range(1, 11)], 
    criterion=["gini", "entropy"]):
    """
    Defines parameter distribution for Random Search,
    converting the lists to np.ndarrays.
    ------------------------------------------------------
    INPUT:
        RandomForestRegressor, etc.)
        max_depth: (list) Depth of tree
        min_samples_split: (list) Sample number to split w/ higher values for
            more sampling, the less likely to create splits based on noise.
        min_samples_leaf: (list) Samples required at each leaf node w/ higher
            value creating more robust predictions (averaging over more samples)
        criterion: (list) Gini or Entropy

    OUTPUT:
        params: (dict) 
    """
    return {
        "max_depth": np.array(max_depth), "min_samples_split": np.array(min_samples_split),
        "min_samples_leaf": np.array(min_samples_leaf), "criterion": criterion
    }

def random_search(
    param_space, 
    n_iterations=N_ITERATIONS
):
    """
    Purpose: ?
    -------------------------------------------------
    INPUT:
        param_space: 

    OUTPUT:
    """
    all_params = []

    for i in range(n_iterations):
        params_sample = {
            "depth": choice(param_space["max_depth"]),
            "split": choice(param_space["min_samples_split"]),
            "leaf": choice(param_space["min_samples_leaf"]),
            "criterion": choice(param_space["criterion"])
        }
        all_params.append(params_sample)

    return all_params

def train_evaluate(parameters, X_train, y_train, X_test, y_test):
    """
    Trains a Decision Tree with given parameters, then evaluates performance.
    --------------------------------------------------------
    INPUT:
        parameters: (dict) Sampled parameters
        X_train: (np.ndarray) Training data
        y_train: (np.ndarray) Training data labels
        X_test: (np.ndarray) Test data features
        y_test: (np.ndarray) Test data labels

    OUTPUT:
        tree: (DecisionTree) Trained Decision Tree model'
        accuracy: (float) Accuracy of model on test data
    """
    tree = dt.DecisionTree(
        max_depth = parameters["depth"],
        min_samples_split=parameters["split"],
        min_samples_leaf=parameters["leaf"],
        criterion=parameters["criterion"]
    )
    
    # Train
    tree.fit(X_train, y_train)

    # Predictions
    y_prediction = tree.prediction(X_test)
    
    # Number of correct predictions divided by the number of test amples
    accuracy = sum(1 for y, p in zip(y_test, y_prediction) if y == p) / len(y_test)
    breakpoint()

    return tree, accuracy

def task1_hyperparam_tuning():
    """
    Task 1: Finds best hyperparameters using random search
    --------------------------------------------------------
    INPUT:
        None

    OUTPUT:
        best_model: ()
        X_test: (np.ndarray) Test data
        y_test: (np.ndarray) Test labels
    """
    print("\nTask 1: Finding best hyperparameter using random search")
    print("="*50)

    # Load and split data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)
    
    # Define parameter space and perform random search
    param_space = define_params()
    # Random Search
    parameter_samples = random_search(param_space)
    # Evaluate all parameter combos
    best_accuracy  = 0
    best_params = None
    best_model = None

    for params in parameter_samples:
        model, accuracy = train_evaluate(params, X_train, y_train, X_test,
                                         y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    print("\nBest Hyperparameters:")
    print(f"Max depth: {best_params['depth']}")
    print(f"Min Samples Split: {best_params['split']}")
    print(f"Min Samples Leaf: {best_params['leaf']}")
    print(f"Criterion: {best_params['criterion']}")
    print(f"Best Accuracy: {best_accuracy:.4f}")

    return best_model, X_test, y_test

def task2_error_analysis(model, X_test, y_test):
    """
    Task 2: Analyze misclassified instances
    ------------------------------------------------
    INPUT:
        model: (?)
        X_test: (list) Test data
        y_test: (list) Test labels

    OUTPUT:
        y_prediction: (list)
    """
    pass

def task3_confusion_martrix(y_test, y_pred):
    """
    Task 3: Calculate and display confusion matrix
    --------------------------------------------------
    INPUT:
        y_test: (list) Test labels
        y_pred: (list) ?

    OUTPUT:
        None
    """
    pass

def task4_regression():
    """
    Task 4: Decision Tree Regression
    ------------------------------------------------------------
    INPUT:

    OUTPUT:

    """
    pass

def main():
    # Task 1: Hyperparameter tuning
    best_model, X_test, y_test = task1_hyperparam_tuning() 

if __name__ == "__main__":
    main()
