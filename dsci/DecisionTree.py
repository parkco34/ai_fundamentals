#!/usr/bin/env python
import numpy as np


class DecisionTree(object):
    def __init__(
        max_depth,
        min_num_samples=2,
        criterion="gini"
    ):        
        self.max_depth = max_depth
        self.min_num_samples = min_num_samples
        self.criterion = criterion

    def parent_entropy(self, y):
        """
        Gets the parent entropy for entire dataset.
        ----------------------------------------
        np.bincount - Counts number of occurences of each value in the array of
        non-negative integers.
        ----------------------------------------
        INPUT:
            y: (np.ndarray)

        OUTPUT:
            entropy: (float)
        """
        total = len(y)
        # Probabilities for the different classes
        probabilities = y.bincount(y) / total
        
        # ENtropy - Ensuring probabilities aren't zero
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def child_entropy(self, X, y, attribute):
        """
        Gets the weighted attribute (child) entropy for attribute value.
        ------------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data
            y: (np.ndarray) Test data.
            attribute: (int) Categorical label (0 or 1)

        OUTPUT:

        """
        total = len(y)
        # Unique values only
        chldren = np.unique(X[attribute])
        weighted_entropy = 0

        logging.debug(f"Entropy for attribute: {attribute}")

        for val in children:
            # Subset target values
            subset_y = y[X[attribute] == val]
            # Total for this subset
            subset_total = len(subset_y)
            # For weighing the subset entropy
            weight = subset_total / total
            subset_entropy = parent_entropy(subset_y)
            weighted_entropy += weight * subset_entropy

            # Log information
            logging.debug(f"{attribute} == {val}")
            logging.debug(f"""Count: {len(subset_y)}  (Pass: pass_count), Fail:
                          {fail_count}""")

            return weighted_entropy

    def parent_gini(self, y):
        """
        Measures the Gini impurity of entire dataset ?
        ------------------------------------------
        INPUT:
            y: (np.ndarray) Target labels.

        OUTPUT:
            weighted_gini: (float) ?
        """
        pass

    def fit(self, X, y):
        pass

    def predict(self):
        pass

    def visualize(self):
        pass







