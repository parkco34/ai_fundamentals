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

def check_for_unique_list_items(lst):
    """
    Checks for the number of unique items in a list, returning the item
    and quantity.
    ------------------------------------------
    INPUT:
        lst: (list) Duh

    OUTPUT:
        counts: (dict)
    """
    counts = {}

    for item in lst:
        
        if item in counts:
            counts[item] += 1

        else:
            counts[item] = 1

    return counts

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
    midpoints = [round((sorted_list[i] + sorted_list[i+1])/2, 5) for i in
                 range(len(sorted_list) - 1)]

    return attribute_df, midpoints

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
    
    # Probabilities for the Benign and MalignantS
    # Maybe turn this shit into a for loop ?
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

def information_gain(subset1, subset2):
    """
    ig = Parent entropy - weighted_child_entropy
    ------------------------------------------------
    INPUT:
        subset1: (pd.Series) Attribute values less than threshold.
        subset2: (pd.Series) Attribute values equal to more than threshold.

    OUTPUT:
        ig: (float) Information gain
    """
    # Entropy of data
    parent_entropy = data_entropy(y)
    
    # Information gain
    return parent_entropy - weighted_child_entropy(subset1, subset2)

def find_best_split(X, data):
    """
    Finds the best attribute to split on for a decision tree, calculates its
    information gain, and tracks information gain for all attributes.
    --------------------------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        data: (pd.DataFrame) Full dataset including target variable

    OUTPUT:
        best_ig: (float) Best information gain
        best_attribute: (str) Attribute with best information gain
        attribute_value_counts: (dict) Dictionary with attribute names as keys
                                and their value counts as values

    NOTES:
        - Prints a message if multiple attributes have the best information gain
        - Assumes existence of helper functions: get_threshold_values,
          split_data_into_two_subsets, and information_gain
        - Recalculates information gain when checking for ties, which may be
          computationally expensive for large datasets
    """
    ig_values = {}
    attribute_ig_info = {}
    best_ig = -1.0
    best_attribute = None

    for attribute in X.columns:
        attribute_df, midpoints = get_threshold_values(data, attribute)
        # Split data into two subsets
        for mid in midpoints:
            subset1 = attribute_df[attribute_df[attribute] < mid]
            susbet2 = attribute_df[attribute_df[attribute] <= mid]
            
            # Information gain calculation
            ig = information_gain(subset1, subset2)

            # Rounding to avoid float precision complications
            ig = round(ig, 5)

            # Populates ig values
            if ig not in ig_values:
                ig_values[ig] = 1

            else:
                ig_values[ig] += 1

            # Initialization
            attribute_ig_info[attribute] = {"ig": ig, "count": None}
            
            # get best information gain
            if ig > best_ig:
                best_ig = ig
                best_attribute = attribute

    # Fill in counts for attributes
    for attribute, info in attribute_ig_info.items():
        info["count"] = ig_values[info["ig"]]
    
    # Filter ot include only duplicate ig values 
    attribute_ig_info = {attr: info for attr, info in attribute_ig_info.items()
                        if info["count"] > 1}

    if ig_values[best_ig] > 1:
        print(f"More than one attribute with best infongain {best_ig}")
        print(f"""Attributes: {[attr for attr, info in attribute_ig_info.items()
             if info['ig']]}""")

    return best_ig, best_attribute, attribute_ig_info

def grow_tree():
    """
    Need to dive into this in detail and come up with my own shit
    """
    pass

def main():
    best_ig, best_attribute, attribute_value_counts = find_best_split(X, data)
    print(f"Best information gain: {best_ig}")
    print(f"Best attribute: {best_attribute}")
    print(f"Attribute value counts: {attribute_value_counts}")

    breakpoint()

if __name__ == "__main__":
    main()
