#!/usr/bin/env python
import numpy as np

class Node:
    def __init__(self):
        self.feature_index = None
        self.is_leaf = False
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

class DecisionTree:

    def __init__(
        self,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        criterion = "gini"
    ):
        """
        max_depth: Max depth of tree
        min_samples_split: Minimum samples required to split internal node
        min_samples_leaf: Minimum number of samples required to be at leaf
        node.
        max_features: Number of features to consider when looking for best
        split
        random_state: Random Seed for reproducibility
        root: Initialized to None
        is_fitted: Flag to indicate whether model has been trained
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.is_fitted = False

    def class_probabilities(y):
        """
        Duh
        ----------------------------------------
        INPUT: 
            y: (np.ndarray) Class labels
        
        OUTPUT:
            probabilities: (list)
        """
        # input validation
        if len(y) == 0:
            print(f"Class probabilities is {len(y)}!")
            return 0

        # Counts number of occurrences for each value in array of non-negative
        # ints
        counter = np.bincount(y)
        probabilities = [count / len(y) for count in counter.values()]
        
        return probabilities

    def impurity(self, y):
        """
        Calculates impurity for split
        ?
        """
        # input validation
        if len(y) == 0:
            print(f"Class probabilities is {len(y)}!")
            return 0

        # Counts number of occurrences for each value in array of non-negative
        # ints
        counter = np.bincount(y)
        probabilities = counter / len(y)

        if self.criterion == "gini":
            # gini
            return 1 - sum(p**2 for p in probabilities)

        else:
            # Entropy
            return -sum(p * np.log2(p) for p in probabilities)

    def information_gain(self, parent, left_child, right_child, criterion="gini"):
        """
        Calculates information gain for a split.
        ------------------------------------------------------
        INPUT:
            parent: (np.ndarray)
            left_child: (np.ndarray)
            right_child: (np.ndarray)
            criterion: (str) Default: "gini"

        OUTPUT:
            gain: (float)
        """
        # Weights of child nodes
        w_left = len(left_child) / len(parent)
        w_right = len(right_child) / len(parent)

        # Parent impurity minus the sum of the products of the chuld impurities
        # and the weights of corresponding children
        if self.criterion == "gini":
            gain = self.get_gini(parent) - (w_left * self.get_gini(left_child)
            + w_right * self.get_gini(right_child))

        else:
            gain = self.entropy(parent) - (w_left * self.entropy(left_child) +
                                          w_right * self.entropy(right_child))

        return gain

    def best_split(self, X, y):
        """
        Finds best split for node through all unique feature values, using them
        as thresholds.  For each feature, for each thrshold value, the data is
        split into right and left branches.  The impurity is then calculated
        between parent and the two child nodes.  THen, information gain is
        determined by how much the imprity is reduced from the split, where we
        keep track of the best information gain.
        -----------------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data
            y: (np.ndarray) Class labels

        OUTPUT:
            best_feature, best_threshold: (tuple of floats)
        """
        best_gain, best_feature, best_threshold = -1, None, None
        
        if isinstance(X, np.ndarray):
            # Corresponding rows and columns
            n_samples, n_features, = len(X), len(X[0])

            for feature in range(n_features):
                # Get unique values for feature
                thresholds = np.unique(X[:, feature])

                for threshold in thresholds:
                    # Split data
                    left_mask = X[:, feature] <= threshold
                    right_mask = ~left_mask

                    if (sum(left_mask) < self.min_samples_leaf or
                        sum(right_mask) < self.min_samples_leaf):
                        continue

                    # Calculate information gain
                    gain = self.information_gain(y, y[left_mask], y[right_mask])

                    # Update best split if this is better
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

                    return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        """
        REcursively build decision tree
        ------------------------------------------------
        INPUT:

        OUTPUT:

        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Create leaf node if stopping criteria met
        if (self.max_depth is not None and depth >= self.max_depth or n_samples
           < self.min_samples_split or
           n_samples < 2 *self.min_samples_leaf or
            n_classes == 1):
            leaf = Node()
            leaf.is_leaf = True
            leaf.value = Counter(y).most_common(1)[0][0] # Most common class

            return leaf

        # Find best split
        best_feature, best_threshold = self.find_best_split(X, y)

        # If no valid split found, create leaf node
        if best_feature is None:
            leaf = Node()
            leaf.is_leaf = True
            leaf.value = Counter(y).most_common(1)[0][0]

        # Create split node
        node = Node()
        node.feature = best_feature
        node.threshold = best_threshold

        # split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # recursively build left and right subtrees
        node.left = self.build_tree(X[left_mask], y[left_mask], depth+1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth+1)
    
        return node

    def fit(X, y):
        """
        Trains the decision tree by finding optimal splits in training data to
        minimize impurity (classification) or error (regression)
        --------------------------------------------------------
        Use:
           - Validates input
           - Sets: n_features, n_classes, is_fitted=True after training, and
           sets is_fitted = True after training.
        r-------------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data
            y: (np.ndarray) Class labels

        OUTPUT:
            None            
        """
        X = np.array(X)
        y = np.array(y)

        # Build tree starting from the root
        self.root = self.build_tree(X, y, depth=0)

        return self

    def traverse_tree(self, x, node):
        """
        Traverses tree to make prediction for one sample,
        returning node value when we get to leaf node.  If 
        ------------------------------------------------------
        INPUT:

        OUTPUT:

        """
        if node.is_leaf:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)

        return self.traverse_tree(x, node.right)

    def preidction(self, X):
        """
        Predict class for X
        ----------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data

        OUTPUT:
            predictions: (np.ndarray)
        """
        X = np.array(X)

        # Make predictions for each sample
        predictions = [self.traverse_tree(x, self.root) for x in X]

        return np.array(predictions)
    
        
# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.3
N_ITERATIONS = 100
RANDOM_STATE = 73
np.random.seed(RANDOM_STATE)
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                   random_state=RANDOM_STATE)

dt = DecisionTree()


breakpoint()
