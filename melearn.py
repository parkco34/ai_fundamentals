#!/usr/bin/env python
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
"""
----------------------------------------
ChatGPT/Claude Tutor --> NOTES
----------------------------------------
1) Understanding Entropy and Information Gain:

    Theory: We’ll first understand what entropy and information gain are and how they are used to split nodes in a decision tree.
    Coding Task: Implement functions to calculate entropy and information gain.

2) Understanding the Gini Index:

    Theory: Learn what the Gini index is and how it is used as an alternative to entropy for splitting nodes.
    Coding Task: Implement a function to calculate the Gini index.

3) Building the Decision Tree:

    Theory: Understand the structure of a decision tree and how recursion is used to build it.
    Coding Task: Implement the recursive function to build the decision tree.

4) Testing and Pruning:

    Theory: Learn about overfitting, testing the decision tree, and pruning techniques.
    Coding Task: Implement basic testing and optional pruning.
"""

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
# Encode target variables
y_encoded = y["Diagnosis"].map({"M": 1, "B": 0})

# 1)
def probs(y):
    """
    Calculate probabilities of attribute(s).
    -----------------------------------------
    INPUT:
        y: (pd.Series) Attribute

    OUTPUT:
        result: (float) Relative Frequencies over total
    """
    # Ensure there's something to calculate
    if len(y) == 0:
        return 0.0

    probability = np.bincount(y) / len(y)
    return probability

def entropy(y):
    """
    The higher the entropy, the more impurities
    ---------------------------------------------
    INPUT:
        y: (pd.Series) Classifications

    OUTPUT:
        entropy: (np.float64)
    """
    # In case empty Series
    if len(y) == 0:
        return 0.0

    return -np.sum(probs(y) * np.log2(probs(y)))

def weighted_entropy(X, y, attribute):
    """
    Proportion of samples with particular value
    ----------------------------------------
    INPUT:
        y: (pd.Series) target values (classes)
        X: (pd.DataFrame) Attributes and values
        attribute: (str) Attribute

    OUTPUT:

    """
    unique_values = X[attribute].unique()

    weighted_entropy = 0
    total_samples = len(y)

    for value in unique_values:
        # Creating mask for samples with this value
        # "mask" is a boolean array for specific rows meeting criteia
        mask = X[attribute] == value
        y_subset = y[mask]
        weight = len(y_subset) / total_samples
        subset_entropy = entropy(y_subset)

        weighted_entropy += weight * subset_entropy

    return weighted_entropy

def info_gain(X, y, attribute):
    """
    We want higher information gain ~ highest reduction in entropy,
    substracting weighed entropy of child nodes from entropy of parent node.
    ------------------------------------------
    INPUT:
        y: (pd.Series) target values (classes)
        X: (pd.DataFrame) Attributes and values
        attribute: (str) Attribute

    OUTPUT:

    """
    parent_entropy = entropy(y)
    ig = parent_entropy - weighted_entropy(X, y, attribute)

    return ig
        
# 2)
def gini_index(y):
    """
    Gini Index - Measure of how mixed a dataset is.
    gini impurity = 1 - ∑p(i)^2
    The lower the index, the better the attribute is for splitting.
    The higher the index, the lower the homogeneity (higher impurity).
    ------------------------------------------
    INPUT:
        y: (pd.Series) Attributes

    OUTPUT:
        impurity: (float) Gini impurity
    """
    return 1 - sum(probs(y)**2)

# 3)
def build_tree(X, y, func=entropy):
    """
    Recursively builds the decision tree.
    -------------------------------------------
    1. Start w/ entire dataset at root node.
    2. Find best split from all attributes, using information gain or gini index.
    3. Once attribute found, create branches (split data into subsets based on attribute's values).
    4. For each subset of data created by the split:
        - If subset is pure (all samples belong to same class) or stopping condition is met, create leaf node.
        - Else, treat subset as new "root" and recursively apply steps 2-4.
    5. Create recursive calls to create subtrees for each branch.
    6. Subtrees combined forming complete decision tree.
    -------------------------------------------
    INPUT:
        y: (pd.Series) target values (classes)
        X: (pd.DataFrame) Attributes and values

    OUTPUT:

    """
    # Most important attribute for spltting first
    pass
 
def testing_pruning():
    pass

breakpoint()
info_gain(X, y_encoded, 'radius1')

