#!/usr/bin/env python
import numpy as np
import logging

class DecisionTree:
    def __init__(self, max_depth=None, min_num_samples=2, criterion="gini"):
        """
        Initialize the DecisionTree classifier.

        Parameters:
        -----------
        max_depth : int, optional (default=None)
            Maximum depth of the tree
        min_num_samples : int, optional (default=2)
            Minimum number of samples required to split a node
        criterion : str, optional (default="gini")
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        """
        self.max_depth = max_depth
        self.min_num_samples = min_num_samples
        self.criterion = criterion

    def get_probabilities(self, array):
        """
        Returns the probability distribution of the given array.

        Parameters:
        -----------
        array : np.ndarray
            Input array of class labels

        Returns:
        --------
        probs : np.ndarray
            Array of probabilities for each unique class
        """
        if len(array) == 0:
            return np.array([])

        total = len(array)
        probs = np.bincount(array) / total
        return probs

    def parent_entropy(self, y):
        """
        Calculate the entropy of the dataset.

        Parameters:
        -----------
        y : np.ndarray
            Array of class labels

        Returns:
        --------
        entropy : float
            Entropy value of the dataset
        """
        if len(y) == 0:
            return 0.0

        probabilities = self.get_probabilities(y)
        # Calculate entropy while avoiding log(0)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def child_entropy(self, X, y, attribute):
        """
        Calculate the weighted entropy for child nodes after splitting on an attribute.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        attribute : int
            Index of the attribute to split on

        Returns:
        --------
        weighted_entropy : float
            Weighted entropy of child nodes
        """
        if len(y) == 0:
            return 0.0

        children = np.unique(X[:, attribute])
        weighted_entropy = 0.0
        total_samples = len(y)

        for val in children:
            # Get indices where feature has this value
            mask = X[:, attribute] == val
            subset_y = y[mask]

            if len(subset_y) == 0:
                continue

            # Weight is the proportion of samples with this feature value
            weight = len(subset_y) / total_samples
            subset_entropy = self.parent_entropy(subset_y)
            weighted_entropy += weight * subset_entropy

        return weighted_entropy

    def parent_gini(self, y):
        """
        Calculate the Gini impurity of the dataset.

        Parameters:
        -----------
        y : np.ndarray
            Array of class labels

        Returns:
        --------
        gini : float
            Gini impurity value
        """
        if len(y) == 0:
            return 0.0

        probabilities = self.get_probabilities(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def child_gini(self, X, y, attribute):
        """
        Calculate the weighted Gini impurity for child nodes after splitting on an attribute.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        attribute : int
            Index of the attribute to split on

        Returns:
        --------
        weighted_gini : float
            Weighted Gini impurity of child nodes
        """
        if len(y) == 0:
            return 0.0

        values = np.unique(X[:, attribute])
        weighted_gini = 0.0
        total_samples = len(y)

        for val in values:
            mask = X[:, attribute] == val
            subset_y = y[mask]

            if len(subset_y) == 0:
                continue

            weight = len(subset_y) / total_samples
            subset_gini = self.parent_gini(subset_y)
            weighted_gini += weight * subset_gini

        return weighted_gini

    def best_split(self, X, y, method="gini"):
        """
        Find the best feature to split on using either Gini impurity or information gain.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        method : str, optional (default="gini")
            The criterion to use for finding the best split

        Returns:
        --------
        best_feat : int
            Index of the best feature to split on
        """
        if len(y) == 0 or X.shape[1] == 0:
            return None

        best_feat = None

        if method == "gini":
            best_criterion = float("inf")
            for feat in range(X.shape[1]):
                gini = self.child_gini(X, y, feat)
                if gini < best_criterion:
                    best_criterion = gini
                    best_feat = feat
        else:
            best_criterion = -float("inf")
            parent_ent = self.parent_entropy(y)
            for feat in range(X.shape[1]):
                child_ent = self.child_entropy(X, y, feat)
                info_gain = parent_ent - child_ent
                if info_gain > best_criterion:
                    best_criterion = info_gain
                    best_feat = feat

        logging.debug(f"Best Feature: {best_feat}")
        logging.debug(f"Best {'Gini' if method == 'gini' else 'information gain'}: {best_criterion}")

        return best_feat

    def plurality_value(self, y, random_state=None):
        """
        Returns the most common output value among a set of examples,
        breaking ties randomly.

        Parameters:
        -----------
        y : np.ndarray
            Array of class labels
        random_state : int, optional (default=None)
            Random state for reproducibility

        Returns:
        --------
        most_common : int
            Most common class label
        """
        if len(y) == 0:
            return None

        values, counts = np.unique(y, return_counts=True)
        max_indices = np.where(counts == counts.max())[0]

        if len(max_indices) == 1:
            return values[max_indices[0]]

        rng = np.random.RandomState(random_state)
        return values[rng.choice(max_indices)]

    def learn_decision_tree(self, X, y, parent_y=None, max_depth=None,
                          min_num_samples=2, current_depth=0, method="gini"):
        """
        Recursively builds the decision tree.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        parent_y : np.ndarray, optional (default=None)
            Parent node's target vector
        max_depth : int, optional (default=None)
            Maximum depth of the tree
        min_num_samples : int, optional (default=2)
            Minimum number of samples required to split a node
        current_depth : int, optional (default=0)
            Current depth in the tree
        method : str, optional (default="gini")
            The criterion to use for finding the best split

        Returns:
        --------
        tree : dict
            The decision tree represented as a dictionary
        """
        if parent_y is None:
            parent_y = y

        # Base cases
        if len(y) == 0:
            return {"class": self.plurality_value(parent_y)}

        if len(np.unique(y)) == 1:
            return {"class": y[0]}

        if X.shape[1] == 0 or (max_depth is not None and current_depth >= max_depth):
            return {"class": self.plurality_value(y)}

        # Find best attribute to split on
        best_feature = self.best_split(X, y, method=method)

        if best_feature is None:
            return {"class": self.plurality_value(y)}

        # Create tree structure
        tree = {
            "feature": best_feature,
            "branches": {}
        }

        # Create branches
        for value in np.unique(X[:, best_feature]):
            mask = X[:, best_feature] == value
            X_subset = np.delete(X[mask], best_feature, axis=1)
            y_subset = y[mask]

            # Recursive call
            subtree = self.learn_decision_tree(
                X_subset,
                y_subset,
                parent_y=y,
                max_depth=max_depth,
                min_num_samples=min_num_samples,
                current_depth=current_depth+1,
                method=method
            )

            tree["branches"][value] = subtree

        return tree

