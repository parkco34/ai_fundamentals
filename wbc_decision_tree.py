#!/usr/bin/env python
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features # float64
y = breast_cancer_wisconsin_diagnostic.data.targets # float64
# Encode target variables as float
y_class = y["Diagnosis"].map({"M": 1, "B": 0}).astype(float)

# 1)
def entropy(y):
    """
    INPUT:
        y: (pd.Series of floats)

    OUTPUT:
        entropy: (float or np.float64)
    """
    # Probailities
    probability = y.value_counts() / len(y)

    # Binary Entropy function
    return -sum(probability * np.log2(probability))

# 2)
def weighted_entropy(X, y, attribute):
    """
    INPUT:
        X: (pd.DataFrame) 
        y: (pd.Series)

    OUTPUT:
        ()
    """
    # Unique values for some reason ????
    unique_values = X[attribute].unique()

    weighted_entropy = 0
    # FINISH


def information_gain(X, y, attribute):
    """
    INPUT:
        X: (pd.DataFrame) 
        y: (pd.Series)

    OUTPUT:
        ()
    """
    parent = entropy(y)
    


#breakpoint()
