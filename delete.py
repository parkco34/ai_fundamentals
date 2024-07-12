#!/usr/bin/env python
"""
Pseduo code from ChatGPT
"""


class DecisionTree:

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # Base cases
        if len(set(y)) == 1:
            return y[0]

        if not X or (self.max_depth and depth >= self.max_depth):
            return self._majority_class(y)

        # Find the best split
        best_feature, best_value = self._best_split(X, y)
        if best_feature is None:
            return self._majority_class(y)

        # Split the dataset
        left_indices, right_indices = self._split(X, best_feature, best_value)
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (best_feature, best_value, left_tree, right_tree)

    def _best_split(self, X, y):
        best_ig = -1
        best_feature, best_value = None, None
        base_entropy = self._entropy(y)

        for feature in range(X.shape[1]):
            values = set(X[:, feature])

            for value in values:
                left_indices, right_indices = self._split(X, feature, value)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_entropy = self._entropy(y[left_indices])
                right_entropy = self._entropy(y[right_indices])
                ig = base_entropy - (
                    len(left_indices) / len(y) * left_entropy +
                    len(right_indices) / len(y) * right_entropy
                )

                if ig > best_ig:
                    best_ig, best_feature, best_value = ig, feature, value

        return best_feature, best_value

    def _split(self, X, feature, value):
        left_indices = [i for i, x in enumerate(X[:, feature]) if x <= value]
        right_indices = [i for i, x in enumerate(X[:, feature]) if x > value]
        return left_indices, right_indices

    def _entropy(self, y):
        proportions = [np.mean(y == c) for c in set(y)]
        return -sum(p * np.log2(p) for p in proportions if p > 0)

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return [self._predict_instance(x, self.tree) for x in X]

    def _predict_instance(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, value, left_tree, right_tree = tree
        if x[feature] <= value:
            return self._predict_instance(x, left_tree)

        else:
            return self._predict_instance(x, right_tree)

