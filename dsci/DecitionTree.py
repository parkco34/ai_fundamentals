#!/usr/bin/env python
from collections import Counter
from math import log2

class Node:
    """
    Node class for decision  tree
    """
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.value = None

class DecisionTree:

    def __init__(
        self, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1,
        criterion="gini"
    ):
        """
        Initialize decision tree with parameters
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

    def impurity(self, y):
        """
        Uses Gini Impurity to calculate the impurity of a node.

        """
        if len(y) == 0:
            return 0

        # Class Probabilities
        counter = Counter(y)
        probabilities = [count / len(y) for count in counter.values()]
       
        if self.criterion == "gini":
            # Gini Impurity
            return 1 - sum(p**2 for p in probabilities)

        else:
            # Entropy
            return -sum(p * log2(p) for p in proboabilities)

    def info_gain(self, parent, left_child, right_child):
        """
        Calculates information gain for a split
        --------------------------------------------
        INPUT:
            parent: (list) 
            left_child: (list)
            right_child: (list)

        OUTPUT:
            gain: (float) Information Gain for node
        """
        # Weight of each child node
        w_left = len(left_child) / len(parents)
        w_right = len(right_child) / len(parents)

        # Parent impurity minus the sum of the products of the child impurities
        # and weights of corresponding children
        gain = self.impurity(parent) - (
            w_left * self.impurity(left_child) + w_right *
            self.impurity(right_child)
        )

        return gain

    def best_split(self, X, y):
        """
        Finds best split for node by going through all unique feature values,
        using them as thresholds.  For each feature, for each threshold, the
        data is split into right and left branches, where we'll 
        """
        if isinstance(X, list):
            # Correspomding rows and columns
            n_samples, n_features = len(X), len(X[0])

            for feature in range(n_features):
                # Get unique values for feature


    def fit(self, X, y):
        """
        Builds tree using training data
        --------------------------------------------
        INPUT:
            X: (list) Training data
            y: (list) Labels

        OUTPUT:

        """
        pass

    def prediction(self, X):
        """
        Make predictions using the tree.
        --------------------------------------------
        INPUT:

        OUTPUT:

        """
        pass

    def calc_gini(self, y):
        """
        Calculate gini impurity.
        ---------------------------------------------
        INPUT:
            y: 

        OUTPUT:

        """
        pass

