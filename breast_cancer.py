#!/usr/bin/env python
"""
Parker, Cory 
AI Fundamentals graduate course at RIT
7/10/24 -- Actual start of project

The Help:
    - https://medium.com/@cristianleo120/master-decision-trees-and-building-them-from-scratch-in-python-af173dafb836
    - Claude 3.5
"""
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
# Encode target variables
y_encoded = y["Diagnosis"].map({"M": 1, "B": 0})

# Data
#print(breast_cancer_wisconsin_diagnostic.metadata) 
# variable information 
#print(breast_cancer_wisconsin_diagnostic.variables) 

class Node:
    """
    Since each node represents a node in the decision tree, this class is
    useful for creating a heircarchical structure that mirrors the tree's
    branching, while storing info such as: attribute, threshold, etc. It also
    encapsulates the info needed at each decision point in the tree, and makes
    it easier to traverse the tree by following the left or right references
    based on the attriubte values of new data. Since it's a recursive process,
    you can create and return Node objects to build up the tree structure. Also
    makes it easier to add functionality (like adding Pruning).
    """

    def __init__(self, attribute=None, threshold=None, left=None, right=None,
                 value=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def proper_split(X, y):
    """
    Selects the highest information gain for the split.
    -------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attributes and their values
        y: (pd.Series) Classifications
    """
    best_attribute = None # No best attribute found initially
    best_info_gain = -1 # Info gain always non-negative

    # Iterate thru attributes
    for attribute in X.columns:
        # Get info gain for attribute and corresponding values
        info_gain = importance(y, X, attribute)
        # Test for greatest info gain to assign to best attribute
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute = attribute

    return best_attribute

def entropy(y):
    """
    => The higher the entropy, the more disorder (impurity).
    Instead of the binary entropy function, I'm using the more genral one as
    it's more flexible, handling multiclasses as well.
    --------------------------------------------------------
    INPUT:
        y: (pd.Series) Classifications (maligant/benign)

    OUTPUT:
        entropy: (float) Between 0 and 1, where the lower the entropy, the
        better.
    """
    # In case an of empty pandas series
    if len(y) == 0:
        return 0.0

    if isinstance(y, pd.Series):
        # Probabilities
        # counts = np.bincount(y)
        # _, counts = np.unique(y, return_counts=True)
        counts = y.value_counts()
        probabilities = counts / len(y)
        # Entropy, using 1*10^-9 in case of p = 0
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))

        return float(entropy)

def importance(y, X, attribute):
    """
    CHOOSING THE MOST IMPORTANT ATTRIBUTE TO TEST FIRST.
    => We want HIGH info gain for splitting.
    --------------------------------
    IG(D_i, f) = Entropy(D_i)  - ∑_(l∈f) (|D_i,l|/|D_i|) * (Entropy(D_i,l)),
     where D_i is the set of instances in the parent node, where 
     i is a particular classification,
     and f is the attribute being looked at for splitting, 
     and l is the different values in the attribute,
     D_i,l is the subset of D_i where attribute f has value l.
    ------------------------------------------------
    INPUT:
        y: (pd.Series) target values (classes)
        X: (pd.DataFrame) Attributes and values
        attribute: (str) Attribute

    OUTPUT:
        info_gain: (float) 
    """
    # Entropy of parent node (D_i)
    parent_entropy = entropy(y)
    # Unique values of attribute
    attr_values = X[attribute].unique()
    
    # Weighted sum of child entropies
    weighted_child_entropy = 0
    for value in attr_values:
        # Creating child dataframe where D_i,l, where attr-value l
        child_y = y[X[attribute] == value] # classification
        print(f"child classification: {type(child_y)}")
        weight = len(child_y) / len(y)
        # Child entropy
        child_entropy = entropy(child_y)
        print(f"child entropy: {type(child_entropy)}")
        # Add weighted child entropy to the sum weighted_child_entropy += weight * child_entropy # Calculate information gain
    info_gain = parent_entropy - weighted_child_entropy

    return info_gain

def make_tree(X, y, depth):
    """
    Builds tree recursively, using node-splitting to create branches,
    including a stopping criteria, where the criteria can be based on the
    purity of the subsets, size of the subsets (min # of samples required to
    split an internal node or at leaf node), or the depth of the tree.
    ------------------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attributes with values
        y: (pd.Series) Classifications
        depth: (int) depth of tree
    """
    pass

def plurality_value(attribute):
    """
    Selects the most common value in the attribute if the attribute is empty,
    depth of tree is the same as max_depth.
    --------------------------------------------------
    INPUT:
        atttribute: (pd.Series) Attribute

    OUTPUT:
        most_common: () Most common value in the attribute
    """
    pass

def decision_tree(examples, attributes, parent_examples):
    """
    Decision Tree learning algorithm.
    ----------------------------------
    INPUT:
        examples: 
        attributes:
        parent_examples:

    OUTPUT:
        tree: 
    """
    # If examples is empty, return most common class from parent
    # examples
    if not examples:
        return plurality_value(parent_examples)

    # if all examples have the same classification, return the calssifcation
    elif len(classes.unique()) == 1:
        return classes[0]

    # if attributes is empty, return the most common class from examples
    elif not attributes:
        return plurality_value(examples)

    # A <- argmax_a in attributes importance(a, examples)
    # tree <- new decision tree with root test 

def pruning():
    """
    Reduces size of decision tree to prevent overfitting (reduced error pruning)
    """
    pass

def regularization():
    """
    Post-pruning to avoid overfitting
    """
    pass

def cross_validation():
    """
    Evaluate performance on unseen data (5-fold)
    """
    pass

df_entropy = entropy(y_encoded)
print(f"\nEntropy for dataset: {df_entropy}\n")

#ig_list = []
#for i in X.columns:
#    print(f"Information gain: {importance(y_encoded, X, i)}")
#    ig_list.append(importance(y_encoded, X, i))

breakpoint()
