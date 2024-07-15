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
def child_entropy(X, y, attribute):
    """
    Because child nodes often have different # of of samples, and so this
    ensures we account for the relative importance of each child node based on
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
    total = len(y)
    counter=0

    for value in unique_values:
        # Mask to use in obtaining subset of corrsponding target values
        mask = X[attribute] == value
        y_subset = y[mask]
        # The # of elements in the subset divided by the total #
        weight = len(y_subset) / total
        # Get subset entropy inorder to multiply it by the weights for the
        # weighted average
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
    return entropy(y) - child_entropy(X, y, attribute)

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
    # All samples in current node belong to same class (pure node)
    if len(y.unique()) == 1:
        return y.unique()[0]

    # max_depth reached: return most common class
    if max_depth is not None and current_depth >= max_depth:
        return max(np.bincount(y))

    # # of samples < minimum # of samples required to split, returning the most
    # common class
    if len(y) < min_samples_split:
        return max(np.bincount(y))

    # Out of attributes to split on
    if len(X.columns) == 0:
        return max(np.bincount(y))

    # find best attribute to split
    best_attribute = find_best_attribute(X, y)

    # Creating tree structure
    tree = {best_attribute: {}}

    # Splitting on best attribute
    for value in X[best_attribute].unique():
        # Create mask for current value
        mask = X[best_attribute] == value
        # Recursively grow tree for subset
        subtree = grow_tree(
            X[mask].drop(columns=[best_attribute]),
            y[mask],
            max_depth,
            min_samples_split,
            current_depth + 1
        )
        
        tree[best_attribute][value] = subtree

    return tree

def predict(tree, sample):
    """
    Predict cancer ???
    ---------------------------------
    INPUT:
        tree: The decision tree
        sample: A single sample to predict

    OUTPUT:
        The predicted class (0 or 1)
    """
    if not isinstance(tree, dict):
        return tree

    attribute = list(tree.keys())[0]
    value = sample[attribute]

    if value in tree[attribute]:
        return predict(tree[attribute][value], sample)
    else:
        # If exact value isn't in tree, find closest one
        closest_value = min(tree[attribute].keys(), key=lambda x: abs(x - value))
        return predict(tree[attribute][closest_value], sample)



# Create the decision tree
decision_tree = grow_tree(X, y_class, max_depth=5, min_samples_split=5)

# Example: predict for a single sample
sample = X.iloc[0]  # Just using the first sample as an example
prediction = predict(decision_tree, sample)
print(f"Prediction for sample: {prediction}")

# Predict for all samples
all_predictions = [predict(decision_tree, sample) for _, sample in X.iterrows()]
#breakpoint()
