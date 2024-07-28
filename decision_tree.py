#!/usr/bin/env python
import pandas as pd
import numpy as np


def read_data(filename):
    """
    Reads data from csv file.
    ----------------------------------------
    INPUT:
        filename: (str) Filename including path.

    OUTPUT:
        (tuple)
    ----------------------------------------
    TEST CASE
        best_attribute = "concavity1"
        best_ig = 0.36082517699473016
    """
    X = pd.read_csv(filename)
    y = X["Diagnosis"].map({"M": 1, "B": 0}).astype(float)
    X.drop("Diagnosis", axis=1,  inplace=True)

    return X, y

def entropy(y):
    """
    Entropy based on class probabilities
    ----------------------------------------
    INPUT:
        y: (pd.Series or np.array) Classifications.

    OUTPUT:
        (float) Entropy.
    """
    total = len(y)
    class_probabilities = y.value_counts() / total

    if len(class_probabilities) == 0:
        return 0

    return -sum(class_probabilities * np.log2(class_probabilities))

def split_data(X, y, attribute, value):
    """
    Partitions data into left and right subsets, based on values.
    ------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        attribute: (str) Column name
        value: (float) 

    OUTPUT:
        left, right: (tuple) Left and right subsets of data
    """
    left_mask = X[attribute] < value
    right_mask = X[attribute] >= value
    left = X[left_mask], y[left_mask]
    right = X[right_mask], y[right_mask]

    return left, right

def weighted_child_entropy(left_y, right_y):
    """
    Calculates the weighted child entropy, where we want weights since the size
    of the number of children will change while traversing the tree.
    ----------------------------------------
    INPUT:
        left_y: () 
        right_y: ()

    OUTPUT:
        weighted_entropy: (float)
    """
    # Total samples if not empty
    total_samples = len(left_y) + len(right_y)
    if total_samples == 0:
        return 0

    left_weight = len(left_y) / total_samples
    right_weight = len(right_y) / total_samples
    
    weighted_entropy = left_weight * entropy(left_y) + (right_weight *
    entropy(right_y))

    return weighted_entropy

def information_gain(X, y, attribute, value):
    """
    Subtract child entropy for each attribute and subtract it from the parent
    entropy (whole dataset).
    ------------------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        attribute: (str) Column name
        value: (float) 

    OUTPUT:
        info_gain: (float) The higher the better the split.
    """
    parent_entropy = entropy(y)
    
    # Getting left and right partitions of data
    left, right = split_data(X, y, attribute, value)
    # Getting the 2nd column of partition, the class labels.
    left_y, right_y = left[1], right[1]

    # Check for appropriate values
    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    # return the Weighted child entropy
    child_entropy = weighted_child_entropy(left_y, right_y)
    info_gain = parent_entropy - child_entropy

    return info_gain, child_entropy

def find_best_split(X, y):
    """
    Finds the best attribute and split value using information gain.
    ------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attributes and their values
        y: (pd.Series) Classifications

    OUTPUT:
        best_attribute, best_ig, best_value, best_child_entropy: (tuple)
    """
    best_attribute = None
    best_ig = -1
    best_value = None
    best_child_entropy = None

    for attribute in X.columns:
        unique_values = sorted(X[attribute].unique())
        mid_points = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]

        for value in mid_points:
            ig, child_entropy = information_gain(X, y, attribute, value)
            if ig > best_ig:
                best_ig = ig
                best_attribute = attribute
                best_value = value
                best_child_entropy = child_entropy

    return best_attribute, best_ig, best_value, best_child_entropy


def grow_tree(X, y, max_depth=None, min_samples_split=2, current_depth=0):
    """
    Builds tree recursively.
    ------------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target data
        max_depth: (int; default: None)
        min_samples_split: (int; default: 2)
        current_depth: (int; default: 0) 

    OUTPUT:
        tree: (dict)
    """
    # -----------------
    # Stopping crtieria
    # -----------------
    # If target classes are the same, return the first element in first index
    if len(y.unique()) == 1:
        return y.iloc[0]

    # If the # of samples is less than the minimum sample split, return the
    # mode - most common class label in order to prevent OVERFITTING.
    if len(y) < min_samples_split:
        return y.mode().iloc[0]

    # If max_depth specified, checks if current depth of tree has been reached
    # or exceeded
    if max_depth is not None and current_depth >= max_depth:
        return y.mode().iloc[0]

    # Find best split (midpoints)
    best_attribute, best_ig, best_value, best_child_entropy = find_best_split(X, y)
    # Debug
    print(f"""
Best attribute: {best_attribute}\n
Best_ig: {best_ig}\n
best value: {best_value}\n
    """)

    # Check for best information gain, returning most common label if
    # irrelevant
    if best_ig <= 0:
        return y.mode().iloc[0]

    # Grow tree
    tree = {best_attribute: {}}

    # Split data
    left, right = split_data(X, y, best_attribute, best_value)
    left_X, left_y = left
    right_X, right_y = right

    # Create subtrees, recursively
    if len(left_y) > 0:
        left_subtree = grow_tree(left_X, left_y, max_depth, min_samples_split, current_depth+1)
        tree[best_attribute][f"<{best_value}"] = left_subtree

    if len(right_y) > 0:
        right_subtree = grow_tree(right_X, right_y, max_depth,
                                  min_samples_split, current_depth+1)
        tree[best_attribute][f">={best_value}"] = right_subtree

    return tree



X, y = read_data("data/example.csv")
tree = grow_tree(X, y)
best_attribute, best_ig, best_value, best_child_entropy = find_best_split(X, y)
left, right = split_data(X, y, best_attribute, best_value)

print(f"Best attribute: {best_attribute}")
print(f"Best information gain: {best_ig}")
print(f"Split value: {best_value}")
print(f"Weighted child entropy: {best_child_entropy}")
print(f"Left subset: \n{left[0]} \nTarget: \n{left[1]}")
print(f"Right subset: \n{right[0]} \nTarget: \n{right[1]}")


#def find_best_split(X, y):
#    """
#    Uses the MIDPOINTS of values instead of the unique values, since it's
#    sightly faster.  It's also better for continuous variables.
#    ------------------------------------------
#    INPUT:
#        X: (pd.DataFrame) Attributes and their values
#        y: (pd.Series) Classifications
#
#    OUTPUT:
#        best_attribute, best_ig, best_value: (tuple)
#    """
#    best_attribute = None
#    best_ig = -1
#    best_value = None
#
#    for attribute in X.columns:
#        # Unique sorted values
#        unique_values = sorted(X[attribute].unique())
#        # Midpoint values
#        mid_points = [(unique_values[i] + unique_values[i+1])/2 for i in
#                      range(len(unique_values) - 1)]
#
#        # Output the midpoints for debugging
#        print(f"Midpoints: {mid_points}") 
#
#        for value in mid_points:
#            # Calculate information gain for threshold value
#            ig = information_gain(X, y, attribute, value)
#            print(f"Value: {value} --> IG({attribute}) = {ig}")
#            if ig > best_ig:
#                best_ig = ig
#                best_attribute = attribute
#                best_value = value
#
#    return best_attribute, best_ig, best_value

breakpoint()
