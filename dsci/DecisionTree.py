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

    def plurality_value(self, parent_examples, random_state=None):
        """
        Returns the most common output value among set of examples, breaking
        ties randomly
        ------------------------------------------------
        INPUT:
            parent_examples: (np.ndarray) 

        OUTPUT:
            basic_bitch: (int) Most common class label, breaking ties randomly
        """
        # If array is empty
        if len(y) == 0:
            return None

        # Get value counts
        values, counts = np.unique(parent_examples, return_counts=True)

        # Find indices with maximum count
        max_indices = np.where(counts == counts.max())[0]
    
        # If single maximum, return it
        if len(max_indices) == 1:
            return values[max_indices[0]]
        
        # Randomly break ties
        rng = np.random.RandomState(random_state)

        return values[rng.choice(max_indices)]

    def learn_decision_tree(self, X, y, max_depth=None, min_num_samples=2,
                  current_dept=0, method="gini"):
        """
        Recursive function that grows the tree, returning the completed tree.
        --------------------------------------------------------
        INPUT:
            X: (np.ndarray) Feature matrix
            y: (np.ndarray) Target vector
            max_depth: (int; default: None)
            min_num_samples: (int; default: 2)
            current_depth: (int; default: 0)
            func: (function; default: find_best_split_ig) Function to find best
            split with; Information gain or Gini Index.

        OUTPUT:
            tree: (dict)
        """
        # If examples empty, return PLURALITY_VALUE(examples)
        if len(y) == 0:
            pass
        
        # If all examples have the same classification

        # if attributes is empty, return PLURALITY_VALUE(examples)
        
        # argmax of most important attribute 


    def clean_tree(self):
        """
        Removes the unneccessary labels for tree in order to correctly
        "process" the predictions
        --------------------------------------------------------
        INPUT:

        OUTPUT:
            
        """
        pass

    def majority_class(self, subtree):
        """
        Handles case when test data contains value not seen during training,
        returning the most common value.
        ------------------------------------------------------
        INPUT:

        OUTPUT:
        """
        pass

    def predict(self, test):
        """
        Predictions with debugging output.
        ------------------------------------------------------
        INPUT:

        OUTPUT:

        """
        pass

    def validate_using_sklearn(self):
        """
    Validates the ( ͡° ͜ʖ ͡°  ) decision tree implementation against
    scikit-learn's
    implementation.
    - Ensures both implementations use entropy as splitting criterion.
    - Uses same max_depth
    - Assumes test_data keys match X.columns
    -----------------------------------------------------------------
    INPUT:
        X: (pd.Dataframe) Training data
        y: (pd.DataFrame) Target labels
        our_tree: (dict) Decision tree
        test_data: (dict) Test instances
        max_depth: (int: optional, default=3) Max depth of tree duh 

    OUTPUT:
        (tuple)
        our_prediction: (str)
        sklearn_prediction: (str)
        accuracy_match: (bool) Whether predictions match
        """
        pass









