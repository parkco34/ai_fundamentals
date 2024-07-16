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

    ---------------------------------------
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

def weighted_child_entropy(X, y, attribute):
    """
    Because child nodes often have different # of of samples, this ensures we account for the relative importance of each child node based on
    its size.
    --------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        attribute: (str)

    OUTPUT:
        (float)
    """
    # Unique values for some reason ????
    unique_values = X[attribute].unique()
    
    weighted_entropy = 0
    total_samples = len(y)
    counter=0

    for value in unique_values:
        # Mask to use in obtaining subset of corrsponding target values
        mask = X[attribute] == value
        y_subset = y[mask]
        # The # of elements in the subset divided by the # of total samples
        weight = len(y_subset) / total_samples
        # Get subset entropy in order to multiply it by the weights for the
        # WEIGHTED AVERAGE
        subset_entropy = entropy(y_subset)
        # Finally
        weighted_entropy += weight * subset_entropy

    return weighted_entropy

def information_gain(X, y, attribute):
    """
    CHOOSING THE MOST IMPORTANT ATTRIBUTE TO TEST FIRST.
    We want higher information gain ~ highest reduction in entropy,
    substracting weighed entropy of child nodes from entropy of parent node.
    ----------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        attribute: (str)

    OUTPUT:
        (float)
    """
    return entropy(y) - weighted_child_entropy(X, y, attribute)

def find_best_attribute(X, y):
    """
    Finds attribute with the highest information gain.
    ------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        
    OUTPUT:
        best_attribute: ?
    """
    best_attribute = None
    best_ig = -1 # Since it has to be positive

    for attribute in X.columns:
        ig = information_gain(X, y, attribute)

        if ig > best_ig:
            best_ig = ig
            best_attribute = attribute

    return best_attribute

def grow_tree(X, y, max_depth=None, min_samples_split=2, current_depth=0):
    """
    Divide and conquer.
    ------------------------------
    INPUT:
        y: (pd.Series) Target data
        y: (pd.Series) Classifications
        max_depth: (int, default: None) Maxmium depth of tree
        min_samples_split: (int, default: 2) Min # of attributes to split

    OUTPUT: """
    # == BASE CASES ==
    # 1) All samples in current node belong to same class (pure node)
    if len(y.unique()) == 1:
        return y.unique()[0]

    # 2) max_depth reached: return MOST COMMON CLASS
    if max_depth is not None and current_depth >= max_depth:
        return max(np.bincount(y))

    # 3) # of samples < minimum # of samples required to split, returning the MOST
    # COMMON CLASS
    if len(y) < min_samples_split:
        return max(np.bincount(y))

    # 4) No attributes to split on
    if len(X.columns) == 0:
        return max(np.bincount(y))

    # find best attribute to split
    best_attribute = find_best_attribute(X, y)

    # Creating tree structure ??
    tree = {best_attribute: {}}
    
    # Splitting on best attribute
    for value in X[best_attribute].unique():
        # Create mask for current value, where the item() part converts
        # np.float64 to float
        mask = X[best_attribute] == value.item()
        # Create new dataframe subset, removing best_attribute
        X_subset = X.drop(columns=[best_attribute])
        # Recursively grow tree for subset
        subtree = grow_tree(
            X_subset,
            y[mask],
            max_depth,
            min_samples_split,
            current_depth + 1
        )
        
        tree[best_attribute][value] = subtree



grow_tree(X, y_class, max_depth=5, min_samples_split=5, current_depth=0)

breakpoint()
