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
            return -sum(p * log2(p) for p in probabilities)

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
        w_left = len(left_child) / len(parent)
        w_right = len(right_child) / len(parent)

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
                thresholds = get_unique_values(X[:, feature])

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
            leaf = Node()
            leaf.is_leaf = True
            leaf.value = Counter(y).most_common(1)[0][0]
            
            return leaf
        
        # Find best split
        best_feature, best_threshold = self.best_split(X, y)

        # If no valid split found, create a leaf node
        if best_feature is None:
            leaf = Node()
            leaf.is_leaf = True
            leaf.value = Counter(y).most_common(1)[0][0]

            return leaf

        # Create decision node
        node = Node()
        node.feature = best_feature
        node.threhsold = best_threshold

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # REcusively build left and right subtrees !
        node.left = self.build_tree(X[left_mask], y[left_mask], depth+1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth+1)

        return node

    def fit(self, X, y):
        """
        Builds tree using training data and labels by establishing the root
        node first, recursively building the branches.
        --------------------------------------------
        INPUT:
            X: (list) Training data
            y: (list) Labels

        OUTPUT:

        """
        self.root = self.build_tree(X, y, depth=0)

        return self

    def traverse_tree(self, x, node):
        """
        Traversing tree, making prediction for a single data sample starting
        from the root node, going down to the leaf node, where at each node, it
        makes a decision based on sample's feature values and threshold stored
        in that node.  This continues until it reaches a leaf node, where it
        outputs the predicted class.
        --------------------------------------------------------
        INPUT:
            X: (list) Training data
            node: () 

        OUTPUT:
        """
        if node.is_leaf == True:
            return node.value

        try:

            if x[node.feature] <= node.threshold:
                return self.traverse_tree(x, node.left)
            
            else:
                return self.traverse_tree(x, node.right)
            
        except Exception as e:
            print(f"Oops... You dun fucked up: {e}")
        
    def prediction(self, X): 
        """
        Make predictions using the tree.
        --------------------------------------------
        INPUT:
            X: (list) Training data

        OUTPUT:
            predictions: (?) ?
        """
        return [self.traverse_tree(x, self.root) for x in X]

    def get_params(self):
        """
        Gets tree parameters.
        ------------------------------------------------------
        INPUT:
            Nada

        OUTPUT:
            (dict) Parameters
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion
        }




