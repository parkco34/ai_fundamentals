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

    def get_probabilities(self, array):
        """
        Returns the probability distribution of the given array.
        -------------------------------------------------------
        INPUT:
            array: (np.ndarray) duh

        OUTPUT:
            probs: (float) Probablities
        """
        total = len(array)
        probs = np.bincount(array) / total

        return probs

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
        # Probabilities for the different classes
        probabilities = get_probabilities(y)
        
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
        # Unique values only
        chldren = np.unique(X[attribute])
        weighted_entropy = 0

        logging.debug(f"Entropy for attribute: {attribute}")

        for val in children:
            # Subset target values
            subset_y = y[X[attribute] == val]
            # getting the probability distribution of subset
            weight = get_probabilities(subset_y)
            subset_entropy = parent_entropy(subset_y)
            weighted_entropy += weight * subset_entropy

            # log information
            logging.debug(f"{attribute} == {val}")
            logging.debug(f"""count: {len(subset_y)}  (pass: pass_count), fail:
                          {fail_count}""")

            return weighted_entropy

    def parent_gini(self, y):
        """
        Gini impurity is a measure of how often a randomly chosen element from
        a set would be incorrectly labeled if it were randomly labeled
        according to the distribution of labels in the subset.  
        Useful in finding the best splits at each node.
        ------------------------------------------
        For a set with k different classes
        Gini Index = 1 - sum_i^k p_i^2, where p_i is the probability of class i
        in the set.
        We calculate the weighted average gini impurity of resulting subset:
            G_split = (n_L/ n) * G_L + (n_R/n) * G_R, where n_L and n_R are
            the number of samples in left/right nodes,
            n = total number of ndoes
            G_L/G_R = Gini impurity of left/right nodes,
            ranging from 0 to 1 - 1/k
            0 = Pure
            1/2 = Max. (classes equally distributed) For binary classification.
        ------------------------------------------
        INPUT:
            y: (np.ndarray) Target labels.

        OUTPUT:
            gini: (float) gini index
        """
        probabilities = get_probabilities(y)
        gini = -np.sum(probabilities**2)

        return gini

    def child_gini(self, X, y, attribute):
        """
        Calculates the weighted gini impurity for children.
        ------------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data
            y: (np.ndarray) Test data.
            attribute: (int) Categorical label (0 or 1)

        OUTPUT:
            weighted_gini: (float) 
        """
        # Get attribute values
        values = np.unique(X[attribute])
        weighted_gini = 0

        for val in values:
            # Subset of attributes
            subset_y = y[X[attribute] == val]
            # Probability distribution of subset
            weight = get_probabilities(subset_y)
            # Subset gini from parent gini of entire dataset
            parent_gini = parent_gini(subset_y)
            weighted_gini += weight * parent_gini

            # log information
            logging.debug(f"{attribute} == {val}")
            logging.debug(f"""count: {len(subset_y)}  (pass: pass_count), fail:
                          {fail_count}""")
            logging.debug(f"{indent}    Gini: {subset_gini:.4f}")
            logging.debug(f"{indent}    Weight: {weight:.4f}")
            logging.debug(f"""{indent}    Weight Gini for '{attribute}' :
                          {weighted_gini:.4f}\n""")
        
        return weighted_gini

    def best_split(self, X, y, method="gini"):
        """
        GINI:
            Finds lowest gini index for the given attribute, looping through
            each attribute value, calculating the weighted gini index, summing
            the product of the weights and the attribute gini indices.
        ------------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data
            y: (np.ndarray) Test data.
            attribute: (int) Categorical label (0 or 1)
            method: (str) Gini or Information gain

        OUTPUT:
            best_feat: (?) Best feature for split
        """
        # initialize best feature
        best_feat = None
        # Initialize with large value since gini ~ 1/info_gain
        best_gini = ("inf")

        for feat in np.unique(X.columns):
            gini = child_gini(X, y, feat)

            # Check for smallest gini index
            if gini < best_gini:
                best_gini = gini
                best_feat = feat

        # Logging
        logging.debug(f"Best Feature: {best_feat}")
        logging.debug(f"Best Gini index: {best_gini}")

        return best_feat

    def grow_tree(self):
        """

        """
        pass



