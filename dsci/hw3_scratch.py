#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random import random, randint, choice
import DecisionTree as dt

# Constants
TEST_SIZE = 0.3
N_ITERATIONS = 100
RANDOM_STATE = 73
np.random.seed(RANDOM_STATE)

def np_to_list(numpy_array):
    """
    Converts numpy.ndarray to lists.
    ------------------------------------
    INPUT:
        numpy_array: (np.ndarray) 

    OUTPUT:
        (list)
    """
    return numpy_array.tolist()

def define_params(max_depth=[i for i in range(1, 21)], min_samples_split=[i for
                                                                          i in range(2, 21)],
                  min_samples_leaf=[i for i in range(1, 11)], criterion=["gini", "entropy"]):
    """
    Defines parameter distribution for Random Search.
    ------------------------------------------------------
    INPUT:
        estimator: (?) Model to use (DecisionTreeClassifier,
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
        "max_depth": max_depth, "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf, "criterion": criterion
    }

def random_search(
    param_space, 
    n_iterations=N_ITERATIONS
):
    """
    Purpose: ?
    -------------------------------------------------
    INPUT:

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
        X_train:
        y_train:
        X_test:
        y_test:

    OUTPUT:
        accuracy: (?) Performance metric for this sample parameter combination
    """
    tree = DecisionTree(
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
    accuracy = sum(1 for y, p in zip(y_test, y_pred) if y == p) / len(y_test)

    return accuracy

def task1_hyperparam_tuning():
    """
    Task 1: Finds best hyperparameters using random search
    --------------------------------------------------------
    INPUT:
        None

    OUTPUT:
        best_model: ()
        X_test: (list) Test data
        y_test: (list) Test labels
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
    parameter_samples = random_search(param_space)
    # Evaluate all parameter combos
    best_accuracy  = 0
    best_params = None
    best_params = None

    for params in parameter_samples:
        model, accuracy = train_evaluate(params, X_train, y_train, X_test,
                                         y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    print("\nBest Hyperparameters:")
    print(f"Max depth: {best_params['depth']}")
    print(f"Min Samples Split: {bet_params['split']}")
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
    print("\nTask 2: Error Analysis")
    print("="*50)
    
    # Make predictions
    y_prediction = model.prediction(X_test)

    # Find misclassified instances
    misclassified = [(i, true, pred) for i, (true, pred) in
                     enumerate(zip(y_test, y_prediction)) if true != pred]

    print("\nMisclassified instancs (Index, True Class, Predicted Class):")
    for idx, true, pred in misclassified:
        print(f"Index: {idx}, True: {true}, Predicted: {pred}")

    return y_prediction

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
    print("\nTask 3: Confusion Matrix")
    print("="*50)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    classes = np.unique(y_test)
    for i in classes:
        # Convert to binary classification for this class
        y_test_binary = (y_test == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()

        print(f"\nClass {i} metrics:")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")

def task4_regression():
    """
    Task 4: Decision Tree Regression
    ------------------------------------------------------------
    INPUT:

    OUTPUT:

    """
    print("\nTask 4: Regression")
    print("="*50)

    # Load Boston Housing dataset
    boston = load_boston()
    X, y = boston.data, boston.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)

    # Train regression
    # ?


def main():
   pass

    

if __name__ == "__main__":
    main()
