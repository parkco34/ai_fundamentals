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

    result = np.bincount(y) / len(y)
    return result


def entropy(y):
    # 1e-9 added incase probability is zero
    return -np.sum(probs(y) * np.log2(probs(y + 1e-9)))

def info_gain(X, y, attribute):
    """
    Choosing best split
    """
    parent = entropy(y)
    # Continue

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
def proper_split(X, y):
    """
    Selects highest info gain for the split
    """
    pass


def build_tree(X, y, func=entropy):
    """
    Recursively builds the decision tree.
    ------------------------------------
    1. Start w/ entire dataset at root node.
    2. Find best split from all attributes, using information gain or gini index.
    3. Once attribute found, create branches (split data into subsets based on attribute's values).
    4. For each subset of data created by the split:
        - If subset is pure (all samples belong to same class) or stopping condition is met, create leaf node.
        - Else, treat subset as new "root" and recursively apply steps 2-4.
    5. Create recursive calls to create subtrees for each branch.
    6. Subtrees combined forming complete decision tree.
    """
    # 1) Root node: Lowest entropy or gini index; highest information gain
    
 


def testing_pruning():
    pass
