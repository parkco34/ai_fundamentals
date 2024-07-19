#!/usr/bin/env python
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


def read_data(filename, test=False):
    """
    Reads data from csv file.
    ----------------------------------------
    INPUT:
        filename: (str) Filename including path.
        test: (bool; default: False) Determines how to obtain data, either for
        testing or not.

    OUTPUT:
        (tuple)
    ----------------------------------------
    TEST CASE
        best_attribute = "concavity1"
        best_ig = 0.36082517699473016
    """
    if not test:
        # fetch dataset
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

        # data (as pandas dataframes)
        X = breast_cancer_wisconsin_diagnostic.data.features # float64
        y = breast_cancer_wisconsin_diagnostic.data.targets # float64
        # Encode target variables as float
        y = y["Diagnosis"].map({"M": 1, "B": 0}).astype(float)
        return X, y

    # Test case
    else:
        X = pd.read_csv(filename)
        y = X["Diagnosis"].map({"M": 1, "B": 0}).astype(float)
        X.drop("Diagnosis", axis=1,  inplace=True)

        return X, y

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

def split_data(X, y, attribute, value):
    """
    Splits the data into two groups based on the given attribute and unique
    value in attribute (column).
    ---------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        attribute: (str) The attribute to split the data on
        value: (float) The value to use as the threshold for the split

    OUTPUT:
        left: (tuple) The left group of data and target values
        right: (tuple) The right group of data and target values
    """
    # Create masks to split the data into left and right groups
    left_mask = X[attribute] < value
    right_mask = X[attribute] >= value
    left = X[left_mask], y[left_mask]
    right = X[right_mask], y[right_mask]

    return left, right

def weighted_child_entropy(left_y, right_y):
    """
    How do we calculate entropy after performing a split ?
    Because child nodes often have different # of of samples, this ensures we account for the relative importance of each child node based on
    its size.
    -----------------------------------------------------------
    H(S|A) = Σ (|S_v| / |S|) * H(S_v)

    In this formula, |S_v| represents the number of instances in subset 
    v, and |S| represents the total number of instances in S.
    ------------------------------------------------------------
    INPUT:
        left_y: (pd.Series) Target data of left child node
        right_y: (pd.Series) Attribute data of right child node

    OUTPUT:
        weighted_entropy: (float) weighted entropy of the kiddos
    """
    total_samples = len(left_y) + len(right_y)
    left_weight = len(left_y) / total_samples
    right_weight = len(right_y) / total_samples
    weighted_entropy = left_weight * entropy(left_y) + (right_weight *
    entropy(right_y))

    return weighted_entropy

def information_gain(X, y, attribute, value):
    """
    ----------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        attribute: (str)
        value: (float) Value used as threshold for split

    OUTPUT:
        (float) Information gain
    """
    # Splitting the data
    left, right = split_data(X, y, attribute, value.item())
    left_y, right_y = left[1], right[1]

    # Parent entropy
    parent = entropy(y)
    # Weighted entropy pf kids
    child_entropy = weighted_child_entropy(left_y, right_y)
    
    return parent - child_entropy

def find_best_attribute(X, y):
    """
    Finds attribute with the highest information gain.
    ------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        
    OUTPUT:
        best_attribute: (str) Column name
    """
    best_attribute = None
    best_ig = -1 # Since it has to be positive
  
    # Loop over unique values to determine best attribute via info gain
    for attribute in X.columns:
        unique_values = X[attribute].unique()

        # Iterate thru unique values to find best value
        for value in unique_values:
            info_gain = information_gain(X, y, attribute, value)
            # Test for best attribute
            if info_gain > best_ig:
                best_ig = info_gain
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

    OUTPUT: 
        tree: (nested dict) Decision tree
    """
    # == BASE CASES ==
    # 1) All samples in current node belong to same class (pure node)
    if len(y.unique()) == 1:
        return y.iloc[0]

    # 2) max_depth reached: return MOST COMMON CLASS
    if max_depth is not None and current_depth >= max_depth:
        return y.mode().iloc[0]

    # 3) # of samples < minimum # of samples required to split, returning the MOST
    # COMMON CLASS
    if len(y) < min_samples_split:
        """ Change this shit ? """
        return max(np.bincount(y))

    # ROOT NODE 
    best_attribute = find_best_attribute(X, y)

    # Creating tree structure ??
    tree = {best_attribute: {}}
    
    # Splitting on best attribute
    for value in X[best_attribute].unique():
        
        # Value as float, not np.float64
        value = value.item()
        # Create mask for current value, where the item() part converts
        mask = X[best_attribute] == value
        # Create new dataframe subset, removing best_attribute
        X_subset = X.drop(columns=[best_attribute])

        # Remove index of value in y
        idx = y[mask].index[0].item() # ?
        y_subset = y.drop(idx, axis=0) # ?
        
        # Recursively grow tree for subset
        subtree = grow_tree(
            X_subset,
            y_subset[mask],
            max_depth,
            min_samples_split,
            current_depth + 1
        )

        tree[best_attribute][value] = subtree

    return tree

def predict(tree, sample):
    """
    Makes prediction for single sample.

    Start at root node. check attribute value in sample.  Based on value, move
    to corresponding branch of tree, otherwise return leaf node.
    ---------------------------------------------
    INPUT:
        tree: (dict) Decision tree
        sample: (pd.DataFrame) Attribute dataframe

    OUTPUT:
        prediction: (float) Predicted class
    """
    # Reached leaf node to be returned
    if isinstance(tree, float):
        return tree

    # Get attribute for current node
    attribute = list(tree.keys())[0]

    # Get value of this attribute in sample
    value = sample[attribute]

    # COnvert to python float if np.float64
    if isinstance(value, np.float64):
        value = value.item()

    # If value exists in our tree, treverse to correpsonding subtree
    if value in tree[attribute]:
        return predict(tree[attribute][value], sample)
    
    else:
        # if value not in tree, reutrn most common class
        return max(tree[attribute], key=lambda x: sum(isinstance(v, float) for
                                                     v in
                                                      tree[attribute][x].values()))

def predict_all(tree, X):
    """
    Make predictions for multiple samples using the decision tree.
    ---------------------------------------------------
    INPUT:
        tree: (nested dict) Decision tree
        X: (pd.DataFrame) Samples to predict

    OUTPUT:
        predictions: ()
    """
    pass



data = read_data("data/partial_dataset.csv", True)
X, y = data[0], data[1]

breakpoint()


#def main():
#    data = read_data("data/partial_dataset.csv", True)
#    X, y = data[0], data[1]
#
#
#if __name__ == "__main__":
