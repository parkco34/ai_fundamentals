#!/usr/bin/env python
import numpy as np
import logging
from typing import Optional, Union, Dict, List, Any

class Node:
    def __init__(self, data=None, children=None, split_on=None,
                 predicted_class=None, is_leaf=False):
        self.data = data
        self.predicted_class = predicted_class
        self.children = children if children is not None else {}
        self.split_on = split_on
        self.is_leaf = is_leaf

class DecisionTree:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_num_samples: int = 2,
        criterion: str = "gini"
    ):
        """
        Initialize DecisionTree classifier

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        min_num_samples : int, default=2
            Minimum number of samples required to split a node
        criterion : str, default="gini"
            The function to measure the quality of a split. Supported criteria are
            'gini' for the Gini impurity and 'entropy' for information gain
        """
        if criterion not in ["gini", "entropy"]:
            raise ValueError("criterion must be either 'gini' or 'entropy'")

        self.root = Node()
        self.max_depth = max_depth
        self.min_num_samples = min_num_samples
        self.criterion = criterion
        self.n_features = None
        self.classes_ = None
        self.feature_types = None

    def _check_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input data"""
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X)
            except:
                raise TypeError("X must be convertible to a numpy array")

        if y is not None:
            if not isinstance(y, np.ndarray):
                try:
                    y = np.array(y)
                except:
                    raise TypeError("y must be convertible to a numpy array")

            if X.shape[0] != y.shape[0]:
                raise ValueError("Number of samples in X and y must match")

        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")

    def _get_feature_type(self, X: np.ndarray, feature_idx: int) -> str:
        """Determine if feature is categorical or numerical"""
        unique_values = np.unique(X[:, feature_idx])
        if len(unique_values) <= 10 and all(isinstance(x, (int, np.integer)) for x in unique_values):
            return "categorical"
        return "numerical"

    def _split_numerical(self, X: np.ndarray, y: np.ndarray, feature: int) -> Dict:
        """Find the best split point for a numerical feature"""
        unique_values = np.sort(np.unique(X[:, feature]))
        best_split = None
        best_criterion = float('inf') if self.criterion == 'gini' else -float('inf')

        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if self.criterion == 'gini':
                left_gini = self.parent_gini(y[left_mask])
                right_gini = self.parent_gini(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / len(y)

                if weighted_gini < best_criterion:
                    best_criterion = weighted_gini
                    best_split = threshold
            else:  # entropy
                info_gain = self.parent_entropy(y) - (
                    np.sum(left_mask) / len(y) * self.parent_entropy(y[left_mask]) +
                    np.sum(right_mask) / len(y) * self.parent_entropy(y[right_mask])
                )
                if info_gain > best_criterion:
                    best_criterion = info_gain
                    best_split = threshold

        return {'type': 'numerical', 'threshold': best_split, 'criterion_value': best_criterion}

    def get_probabilities(self, array: np.ndarray) -> np.ndarray:
        """Calculate probability distribution of classes"""
        if len(array) == 0:
            return np.array([])

        unique_classes = np.unique(array)
        probs = np.zeros(len(self.classes_))
        counts = np.bincount(array, minlength=len(self.classes_))
        probs = counts / len(array)
        return probs

    def parent_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of the target distribution"""
        if len(y) == 0:
            return 0.0
        probabilities = self.get_probabilities(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def parent_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity of the target distribution"""
        if len(y) == 0:
            return 0.0
        probabilities = self.get_probabilities(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def best_split(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Find the best feature to split on"""
        if len(y) == 0 or X.shape[1] == 0:
            return None

        best_split_info = {
            'feature': None,
            'type': None,
            'threshold': None,
            'criterion_value': float('inf') if self.criterion == 'gini' else -float('inf')
        }

        for feature in range(X.shape[1]):
            if self.feature_types[feature] == 'numerical':
                split_info = self._split_numerical(X, y, feature)
            else:  # categorical
                split_info = {
                    'type': 'categorical',
                    'criterion_value': (
                        self.child_gini(X, y, feature) if self.criterion == 'gini'
                        else self.parent_entropy(y) - self.child_entropy(X, y, feature)
                    )
                }

            if (self.criterion == 'gini' and split_info['criterion_value'] < best_split_info['criterion_value']) or \
               (self.criterion == 'entropy' and split_info['criterion_value'] > best_split_info['criterion_value']):
                best_split_info.update(split_info)
                best_split_info['feature'] = feature

        return best_split_info if best_split_info['feature'] is not None else None

    def plurality_value(self, y: np.ndarray, random_state: Optional[int] = None) -> Any:
        """Return most common class label"""
        if len(y) == 0:
            return None

        values, counts = np.unique(y, return_counts=True)
        max_indices = np.where(counts == counts.max())[0]

        if len(max_indices) == 1:
            return values[max_indices[0]]

        rng = np.random.RandomState(random_state)
        return values[rng.choice(max_indices)]

    def learn_decision_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_y: Optional[np.ndarray] = None,
        current_depth: int = 0
    ) -> Node:
        """Build the decision tree recursively"""
        if parent_y is None:
            parent_y = y

        node = Node(data={'X': X, 'y': y})

        # Check stopping criteria
        if len(y) == 0:
            node.predicted_class = self.plurality_value(parent_y)
            node.is_leaf = True
            return node

        if len(np.unique(y)) == 1:
            node.predicted_class = y[0]
            node.is_leaf = True
            return node

        if X.shape[1] == 0 or (self.max_depth is not None and current_depth >= self.max_depth):
            node.predicted_class = self.plurality_value(y)
            node.is_leaf = True
            return node

        # Find best split
        split_info = self.best_split(X, y)
        if split_info is None:
            node.predicted_class = self.plurality_value(y)
            node.is_leaf = True
            return node

        feature = split_info['feature']
        node.split_on = feature

        if split_info['type'] == 'numerical':
            threshold = split_info['threshold']
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            node.children = {
                'left': self.learn_decision_tree(
                    X[left_mask], y[left_mask], y, current_depth + 1
                ),
                'right': self.learn_decision_tree(
                    X[right_mask], y[right_mask], y, current_depth + 1
                )
            }
            node.threshold = threshold
        else:  # categorical
            for value in np.unique(X[:, feature]):
                mask = X[:, feature] == value
                node.children[value] = self.learn_decision_tree(
                    X[mask], y[mask], y, current_depth + 1
                )

        return node

    def _predict_single(self, x: np.ndarray, node: Node) -> Any:
        """Make prediction for a single sample"""
        if node.is_leaf:
            return node.predicted_class

        feature = node.split_on
        if hasattr(node, 'threshold'):  # numerical split
            if x[feature] <= node.threshold:
                return self._predict_single(x, node.children['left'])
            return self._predict_single(x, node.children['right'])
        else:  # categorical split
            value = x[feature]
            if value not in node.children:
                # Handle unseen categorical values
                return self.plurality_value(node.data['y'])
            return self._predict_single(x, node.children[value])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Train the decision tree"""
        self._check_input(X, y)

        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)

        # Determine feature types
        self.feature_types = [self._get_feature_type(X, i) for i in range(X.shape[1])]

        # Build the tree
        self.root = self.learn_decision_tree(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X"""
        self._check_input(X)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X must have {self.n_features} features")

        return np.array([self._predict_single(x, self.root) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X"""
        predictions = self.predict(X)
        probabilities = np.zeros((len(X), len(self.classes_)))

        for i, prediction in enumerate(predictions):
            class_idx = np.where(self.classes_ == prediction)[0][0]
            probabilities[i, class_idx] = 1.0

        return probabilities

