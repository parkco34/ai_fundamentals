#!/usr/bin/env python
import numpy as np
from collections import Counter

class Node:
    """Node class for decision tree"""
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.value = None

class DecisionTree:
    """Decision Tree Classifier"""

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion='gini'):
        """Initialize the decision tree"""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None

    def _calculate_impurity(self, y):
        """Calculate impurity (gini or entropy) of a node"""
        if len(y) == 0:
            return 0

        # Calculate class probabilities
        counter = Counter(y)
        probs = [count / len(y) for count in counter.values()]

        if self.criterion == 'gini':
            # Gini impurity
            return 1 - sum(p * p for p in probs)
        else:
            # Entropy
            return -sum(p * np.log2(p) for p in probs)

    def _calculate_information_gain(self, parent, left_child, right_child):
        """Calculate information gain for a split"""
        # Weight of each child node
        w_left = len(left_child) / len(parent)
        w_right = len(right_child) / len(parent)

        # Calculate gain
        gain = self._calculate_impurity(parent) - (
            w_left * self._calculate_impurity(left_child) +
            w_right * self._calculate_impurity(right_child)
        )

        return gain

    def _find_best_split(self, X, y):
        """Find the best split for a node"""
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            # Get unique values for the feature
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # Skip if split doesn't meet minimum samples
                if (sum(left_mask) < self.min_samples_leaf or
                    sum(right_mask) < self.min_samples_leaf):
                    continue

                # Calculate information gain
                gain = self._calculate_information_gain(
                    y,
                    y[left_mask],
                    y[right_mask]
                )

                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Create a leaf node if stopping criteria are met
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            n_classes == 1):

            leaf = Node()
            leaf.is_leaf = True
            leaf.value = Counter(y).most_common(1)[0][0]  # Most common class
            return leaf

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # If no valid split found, create a leaf node
        if best_feature is None:
            leaf = Node()
            leaf.is_leaf = True
            leaf.value = Counter(y).most_common(1)[0][0]
            return leaf

        # Create split node
        node = Node()
        node.feature = best_feature
        node.threshold = best_threshold

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        """Build the decision tree"""
        # Convert input to numpy array if needed
        X = np.array(X)
        y = np.array(y)

        # Build the tree starting from the root
        self.root = self._build_tree(X, y, depth=0)

        return self

    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction for one sample"""
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        """Predict class for X"""
        # Convert input to numpy array if needed
        X = np.array(X)

        # Make predictions for each sample
        predictions = [self._traverse_tree(x, self.root) for x in X]

        return np.array(predictions)

    def get_params(self):
        """Get tree parameters"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion
        }

# Example usage:
if __name__ == "__main__":
    # Create some sample data
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    y = np.array([0, 0, 1, 1])

    # Create and train the tree
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)

    # Make predictions
    predictions = tree.predict(X)
    print("Predictions:", predictions)

