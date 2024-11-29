# Predcited class labels!/usr/bin/env python
from typing import Optional, Union, Dict, List, Any
import logging
import numpy as np


class Node:
    def __init__(self, data=None, children=None, split_on=None,
                 predicted_class=None, is_leaf=False):
        self.data = data
        self.predicted_class = predicted_class
        self.children = children if children is not None else []
        self.split_on = split_on
        self.is_leaf = is_leaf


class DecisionTree(object):
    def __init__(
        self,
        max_depth: OPtional[int] = None,
        min_num_samples: int = 2,
        criterion: str = "gini"
    ):        
        """
        Initialize DecisionTree classifier
        ------------------------------------------
        INPUT:
            max_depth: (int, optional) (default=None)
            min_num_samples: (int) (default=2) Minimum number of samples
            required to split a node
            criterion: (str) (default='gini') Supported criteria are 'gini' for
            the Gini IMpurity and 'entropy' for the information gain; function
            to measure the quality of the split
        """
        # Check for proper criterion
        if criterion not in ["entropy", "gini"]:
            raise ValueError("Criterion must be either 'gini' or 'entropy'")
        # Invoke Node class
        self.root = Node()
        self.max_depth = max_depth
        self.min_num_samples = min_num_samples
        self.criterion = criterion
        self.n_features = None
        self.classes_ = None
        self.feature_types = None

    def _check_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data
        """
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X)

            except:
                raise TypeError("X must be convertible to numpy array")

        if y is not None:
            if not isinstance(y, np.ndarray):
                try:
                    y = np.array(y)

                except:
                    raise TypeError("y must be convertible to numpy array")

            if X.shape[0] != y.shape[0]:
                raise TypeError("Number of samples in X and y must match!")

        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")

    def _get_feature_type(self, X: np.ndarray, feature_idx: int) -> str:
        """
        Determines if features is categorical or numerical.
        ------------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data
            feature_idx: (int) ?

        OUTPUT:
            feature_type: (str) Categorical or numerical
        """
        unique_values = np.unique(X[:, feature_idx])

        if len(unique_values) <= 10 and all(instance(x, (int, np.integer)) for
                                            x in unique_values):
            return "categorical"

        return "numerical"

    def split_numerical(self, X: np.ndarray, y: np.ndarray, feature: int):
        """
        Finds best split point for numerical feature.
        ------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data
            y: (np.ndarray) Target labels
            feature: (?) Individual feature
        """
        pass

    def get_probabilities(self, array):
        """
        Returns the probability distribution of the given array.
        -------------------------------------------------------
        INPUT:
            array: (np.ndarray) duh

        OUTPUT:
            probs: (np.ndarray) Array of probabilities for each unique class
        """
        # Ensure we dont have an empty array, returning an empty array if so
        if len(array) == 0:
            return np.array([])

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
        # Ensure we have a populated array
        if len(y) == 0:
            return 0.0

        # Probabilities for the different classes
        probabilities = self.get_probabilities(y)
        
        # ENtropy - Ensuring probabilities avoiding log(0)
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
            weighted_entropy: (float) Weighted entropy of child nodes
        """
        # ensure non-empty array
        if len(y) == 0:
            return 0.0

        # Unique values only
        children = np.unique(X[:, attribute])
        weighted_entropy = 0.0
        total_samples = len(y)

        logging.debug(f"Entropy for attribute: {attribute}")

        for val in children:
            # Get indices where feature has this value
            mask = X[:, attribute] == val 
            # Subset target values
            subset_y = y[mask]

            # Ensure length of subset_y array is non-empty
            if len(subset_y) == 0:
                continue

            # Weight, parent entropy and weighted entropy
            weight = len(subset_y) / total_samples
            subset_entropy = self.parent_entropy(subset_y)
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
        probabilities = self.get_probabilities(y)
        gini = 1 - np.sum(probabilities**2)

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
        if len(y) == 0:
            return 0.0

        # Get attribute values
        values = np.unique(X[:, attribute])
        weighted_gini = 0.0
        total_samples = len(y)

        for val in values:
            # Mask
            mask = X[:, attribute] == val
            # Subset of attributes
            subset_y = y[mask]

            if len(subset_y) == 0:
                continue

            weight = len(subset_y) / total_samples
            # Subset gini from parent gini of entire dataset
            subset_gini = self.parent_gini(subset_y)
            weighted_gini += weight * subset_gini
        
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
            best_feat: (int) Index of the best feature to split on
        """
        if len(y) == 0 or X.shape[1] == 0:
            return None

        # initialize best feature
        best_feat = None

        if method == "gini":
            # Initialize with large value since gini ~ 1/info_gai
            best_gini = float("inf")
            
            # For each attribute, comapre some stuff
            for feat in range(X.shape[1]):
                gini = self.child_gini(X, y, feat)

                # Check for smallest gini index
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat

            best_criterion = best_gini

        else:
            # Entropy for information gain
            best_criterion = -float("inf")
        
            for feat in range(X.shape[1]):
                # calculate information gain
                parent_entropy = self.parent_entropy(y)
                child_entropy = self.child_entropy(X, y, feat)
                info_gain = parent_entropy - child_entropy

                # Update if we find larger information gain
                if info_gain > best_criterion:
                    best_criterion = info_gain
                    best_feat = feat

        # Logging
        logging.debug(f"Best Feature: {best_feat}")
        logging.debug(f"""Best {'Gini' if method == 'gini' else 'information gain'}: {best_criterion}""")

        return best_feat

    def plurality_value(self, parent_examples, random_state=None):
        """
        Returns the most common output value among set of examples, breaking
        ties randomly
        ------------------------------------------------
        INPUT:
            parent_examples: (np.ndarray) Target labels
            random_state: (int) (default=None) Random state that introduces
            reproducabiliy

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

    def learn_decision_tree(
        self, 
        X, 
        y, 
        parent_y=None, 
        max_depth=None, 
        min_num_samples=2,
        current_dept=0, 
        method="gini"
    ):
        """
        Recursive function that grows the tree, returning the completed tree.
        --------------------------------------------------------
        INPUT:
            X: (np.ndarray) Feature matrix
            y: (np.ndarray) Target vector
            parent_y: (np.ndarray) Parent node's target vector (for empty nodes)
            max_depth: (int; default: None)
            min_num_samples: (int; default: 2)
            current_depth: (int; default: 0)
            func: (function; default: find_best_split_ig) Function to find best
            split with; Information gain or Gini Index.

        OUTPUT:
            tree: (dict)
        """
        # If first call, set parent_y to current y
        if parent_y is None:
            parent_y = y

        # If examples empty, return PLURALITY_VALUE(parent_examples)
        if len(y) == 0:
            return {"class": self.plurality_value(parent_y)}
        
        # If all examples have the same classification
        if len(np.unique(y)) == 1:
            return {"class": y[0]}
        
        # if attributes is empty or max depth reached
        if X.shape[1] == 0 or (max_depth is not None and current_depth >=
                               max_depth):
            return {"class": self.plurality_value(y)}

        # Find best attributes using existing best_split method
        best_feature = self.bes_split(X, y, method=method)

        if best_feature is None:
            return {"class": self.plurality_value(y)}

        # Create tree structure
        tree = {
            "feature": best_feature,
            "branches": {}
        }

        # For each value of the best feature
        for value in np.unique(X[:, best_feature]):
            # Create mask for feature value
            mask = X[:, best_feature] == value

            # Get subset of data excluding the used feature
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

            # Add branch to tree
            tree["branches"][value] = subtree

        return tree

    def traverse_tree(self, x, tree):
        """
        Traverse decision tree for a single sample to make prediction
        ----------------------------------------------------------------
        INPUT:

        OUTPUT:

        """
        # Base case if we've hit a leaf node, do weird shit
        if "class" in tree:
            return tree["class"]

        # Get the feature index we're splitting on at this node
        feature = tree["feature"]
        # Get the values for this feature in our input sample
        value = x[feature]

        # Handle case whre we see a feature value not present in traiing data
        if value not in tree["branches"]:
            # Collect all class values from current value
            classes = []
            for subtree in tree["branches"].values():
                
                if "class" in subtree:
                    classes.append(subtree["class"])

            # If we found any classes, return most common
            if classes:
                # Get most frequent class
                return max(set(classes), key=classes.count)
            
            # If no classes found return the first available  class
            return list(tree["branches"].values())[0]["class"]

        # Recursively traverse proper branch
        return self.traverse_tree(x, tree["branches"][value])

    def predict(self, X):
        """
        Predict class probabilities for samples in X
        -------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data

        OUTPUT:
            predcitions: (np.ndarray) Predcited class labels
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

        if len(X.shape) != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X mnust have shape (n_samples, {self.n_features})")

        # Make predictions for each sample
        predictions = np.array([self._traverse_tree(x, self.tree) for x in X])

        return predictions
        
    def predict_prob(self, X):
        """
        Predict class probabilities for samples in X.
        ------------------------------------------------
        INPUT:
            X: (np.ndarray) training data

        OUTPUT:
            probabilities: (np.ndarray) Class Probabilities of shape
            (n_samples, n_classes)
        """
        if not hasattr(self, "classes_"):
            self.classes_ = np.unique(self.predict(X))

        # Get predictions
        predictions = self.predict(X)

        # Calculate probabilities using get_probabilities method
        probabilities = np.zeros(len(X), len(self.classes_))
        for i, class_label in enumerate(self.classes_):
            probabilities[:, i] = (predictions == class_label).astype(float)

        return probabilities
        
    def fit(self, X, y):
        """
        Train the decision tree classifier
        ------------------------------------------------
        INPUT:
            X: (np.ndarray) Training feature matrix of shape (n_samples,
            n_features)
            y: (np.ndarray) Target labels of shape (n_samples)

        OUTPUT:
        """
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays!")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")

        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
            
        # Store number of features
        self.n_features = X.shape[1]

        # Train tree using specific criterion
        self.tree = self.learn_decision_tree(
            X,
            y,
            max_depth=self.max_depth,
            method=self.criterion
        )

        return self

    def validate_using_sklearn(self):
        """
    Validates the ( ͡° ͜ʖ ͡°  ) decision tree implementation against
    scikit-learn's
    implementation.
    - Ensures both implementations use entropy as splitting criterion.
    - Uses same max_depth
    - Assumes test_data keys match X.shape[1]
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








dt = DecisionTree()
breakpoint()

