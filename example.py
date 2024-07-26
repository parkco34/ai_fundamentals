#!/usr/bin/env python
import numpy as np
import pandas as pd
from math import log2

"""
Manually implementing decision tree algorithm to part of the dataset in
`example.csv`.
"""

data = pd.read_csv("data/example.csv")
# Encode target attribute values to be 0 or 1
data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})
X, y = data.iloc[:, :-1], data.iloc[:, -1]

def data_entropy(y):
    """
    INPUT:
        y: (pd.Series)

    OUTPUT:
        entropy: (float)
    """
    total = len(y)
    probabilities = y.value_counts() / total # Outputs pd.Series of probs

    return -sum([p * log2(p) for p in probabilities if p > 0])

def get_threshold_values(data, attribute):
    """
    INPUT:
        data: (pd.DataFrame)
        attribute: (str)

    OUTPUT:
        attribute_df: (pd.DataFrame), midpoints: (list)
    """
    # Create 2D dataframe for atribute
    attribute_df = data[[attribute, data.columns[-1]]] 
    # Sort attribute values
    sorted_list = list(sorted(attribute_df[attribute]))
    # Get midpoint values
    midpoints = [round((sorted_list[i] + sorted_list[i+1])/2, 2) for i in
                 range(len(sorted_list) - 1)]

    return attribute_df, midpoints

def split_data_into_two_subsets(attribute_df, midpoints):
    """
    INPUT:
        attribute_df: (pd.DataFrame) 2D dataframe of attribute and target data
        midpoints: (list)

    OUTPUT:
        subset1, subset2: (tuple)
    """    
    attribute = attribute_df.columns[0] # Get attribute name

    for mid in midpoints:
        subset1 = attribute_df[attribute_df[attribute] < mid]
        subset2 = attribute_df[attribute_df[attribute] >= mid]

    return subset1, subset2 # goes to weighted_child_entropy

def weighted_child_entropy(subset1, subset2):
    """
    INPUT:
        subset1: (pd.Series) Attribute values less than threshold.
        subset2: (pd.Series) Attribute values equal to more than threshold.

    OUTPUT:
        child_entropy: (float)
    """
    total1 = len(subset1)
    total2 = len(subset2)
    total = total1 + total2

    s1_values = subset1["Diagnosis"].value_counts()
    s2_values = subset2["Diagnosis"].value_counts()
    
    # Probabilities for the Benign and Malignant
    P_B1 = float(s1_values.get(0, 0) / total1 if total1 != 0 else 0)
    P_M1 = float(s1_values.get(1, 0) / total1 if total1 != 0 else 0)
    P_B2 = float(s2_values.get(0, 0) / total2 if total2 != 0 else 0)
    P_M2 = float(s2_values.get(1, 0) / total2 if total2 != 0 else 0)
    
    # Entropy for each subset
    entropy1 = -sum(p * log2(p) for p in [P_B1, P_M1] if p > 0)
    entropy2 = -sum(p * log2(p) for p in [P_B2, P_M2] if p > 0)

    # Weighted average of the entropy of subsets
    weighted_entropy = (total1 / total) * entropy1 + ((total2 / total) *
    entropy2)

    return weighted_entropy

def information_gain(X, y, attribute, value):
    """
    ig = Parent entropy - weighted_child_entropy
    ------------------------------------------------
    INPUT:

    OUTPUT:

    """
    parent_entropy = data_entropy(y)
    # ???  Basically done.

def grow_tree():
    """
    Need to dive into this in detail and come up with my own shit
    """
    pass


# Example:
rad_df, midpoints = get_threshold_values(data, "radius1")
subset1, subset2 = split_data_into_two_subsets(rad_df, midpoints)
kid_entropy = weighted_child_entropy(subset1, subset2)

breakpoint()
