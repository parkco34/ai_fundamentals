#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Constants
TEST_SIZE = 0.3
RANDOM_STATE = 73
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

def random_search(param_space):
    """

    """

breakpoint()
