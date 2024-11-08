#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random import random, randint, choice
import DecisionTree as dt

# Constants
TEST_SIZE = 0.3
RANDOM_STATE = 73
N_ITERATIONS = 100
np.random.seed(RANDOM_STATE)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                   random_state=RANDOM_STATE)

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

def define_params(max_depth=range(1, 21), min_samples_split=range(2, 21),
                  min_samples_leaf=range(1, 11), criterion=["gini", "entropy"]):
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
    Not Simulated Annealing, apparently...
    Purpose: ?

    Example usage:
    ------------------------------------------------------------
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
    accuracy = sum(y == prediction / y_test) / len(y_test)

    return accuracy


# Example usage
#X = np_to_list(X_train)
#y = np_to_list(y_train)
#params = define_params()
#rs = random_search(params)


#breakpoint()
