#!/usr/bin/env python
from collections import Counter
from math import log2

def get_unique_values(feature):
    """
    Gets the unique values of feature while preserving the order.
    -------------------------------------------------------------
    INPUT:
        feature: (list)

    OUTPUT:
        unique_values: (list)
    """
    return list(dict.fromkeys(feature))


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
        using them as thresholds.  For each feature, for each threshold value, the
        data is split into right and left branches.  The impurity is then
        calculated between parent and the two child nodes.  Following this, the
        information gain is determined how much the impurity is reduced from
        the split, where we keep track of the best information gain.
        -----------------------------------------------------------
        INPUT:
            X: (list) Training data
            y: (list) Target labels

        OUTPUT:
            best_feature, best_threshold: (tuple of floats) 
        """
        best_gain = -1
        best_feature, best_threshold = None, None

        if isinstance(X, list):
            # Corresponding rows and columns
            n_samples, n_features = len(X), len(X[0])

            for feature in range(n_features):
                # Get unique values for feature
                thresholds = np.unique(X[:, feature])

                for threshold in thresholds:
                    # Split data
                    left_mask = X[:, feature] <= threshold
                    right_mask = ~left_mask

                    # Skip if split doesn't meet min samples
                    if (sum(left_mask) < self.min_samples_leaf or
                    sum(right_mask) < self.min_samples_leaf):
                        continue

                    # Calculate information gain
                    gain = self.info_gain(y, y[left_mask], y[right_mask])

                    # update best split
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        """
        Recursively builds tree.
        --------------------------------------------------------
        INPUT:

        OUTPUT:

        """
        # Get shape values for iteration and classes
        n_samples, n_features = len(X), len(X[0])
        n_classes = len(get_unique_values(y))

        # Create a leaf node if stopping criteria are met
        if (
            self.max_depth is not None 
            and depth >= self.max_depth 
            or n_samples < self.min_samples_split 
            or n_samples < 2 * self.min_samples_leaf
            or n_classes == 1
        ):
            
            return leaf
        
        # Find best split
        best_feature, best_threshold = self.best_split(X, y)

        # If no valid split found, create a leaf node
        if best_feature is None:
            leaf = Node()
            leaf.is_leaf = True
            leaf.value = Counter(y).most_common(1)[0][0]

            return leaf

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # REcusively build left and right subtrees
        node.left = self.build_tree(X[lef_mask], y[left_mask], depth+1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth+1)

        return node

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



