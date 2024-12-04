# Predcited class labels!/usr/bin/env python
from typing import Optional, Union, Dict, List, Any
import pandas as pd
import logging
import numpy as np


class Node:
    def __init__(self, data=None, children=None, split_on=None,
                 predicted_class=None, is_leaf=False):
        self.data = data
        self.predicted_class = predicted_class
        self.children = children if children is not None else {}
        self.split_on = split_on
        self.is_leaf = is_leaf


class DecisionTree(object):
    def __init__(
        self,
        max_depth: Optional[int] = None,
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
        # why ?
        unique_values = np.unique(X[:, feature_idx])

        if len(unique_values) <= 10 and all(isinstance(x, (int, np.integer)) for
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
        # Get sorted unique values
        unique_values = np.sort(np.unique(X[:, feature]))
        best_split = None
        best_criterion = float("inf") if self.criterion == "gini" else -float("inf")

        # For each pair of consecutive values, split between them nuts
        for i in range(len(unique_values)-1):
            threshold = (unique_values[i] + unique_values[i+1]) / 2

            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if self.criterion == "gini":
                # Calculate weights gini impurity
                left_gini = self.parent_gini(y[left_mask])
                right_gini = self.parent_gini(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                weighted_gini = (n_left * left_gini + n_right * right_gini) / \
                len(y)

                # Compare gini shit
                if weighted_gini < best_criterion:
                    best_criterion = weighted_gini
                    best_split = threshold

            else:
                info_gain = self.parent_entropy(y) - (
                    np.sum(left_mask) / len(y) * self.parent_entropy(y[left_mask]) +
                    np.sum(right_mask) / len(y) * self.parent_entropy(y[right_mask])
                )
                if info_gain > best_criterion:
                    best_criterion = info_gain
                    best_split = threshold

        return {'type': 'numerical', 'threshold': best_split, 'criterion_value': best_criterion}

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
        
        unique_classes = np.unique(array)
        probs = np.zeros(len(self.classes_))
        counts = np.bincount(array, minlength=len(self.classes_))
        probs = counts / len(array)

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

    def _predict_single(self, x: np.ndarray, node: Node) -> Any:
        """
        Makes prediction for single sample, traversing the tree recursively,
        until leaf node is reached...
        For non-leaf nodes, get feature to split on.
        for numerical values (threshold),
        if x[feature] <= threshold -> go left
else -> go right

        For categorical features: 
            if feature value exists in children -> follow that path 
else -> return most common class from node's data
        ------------------------------------------------
        INPUT:
            x: traning data sample
            node: (Node) ?
        """
        if node.is_leaf:
            return node.predicted_class
        
        # ?
        feature = node.split_on

        if hasattr(node, "threshold"):
            return (self._predict_single(x, node.children["left"])
                   if x[feature] <= node.threshold
                   else self._predict_single(x, node.children["right"]))

        # Obtain value for that feature
        value = x[feature]
        if value not in node.children:
            return self.plurality_value(node.data["y"])

        return self._predict_single(x, node.children[value])
    

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for samples in X by applying the trained
        decisionn tree
        -----------------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data

        OUTPUT:
            ?
        """
        # Input validation
        self._check_input(X)
        # Check if ?
        if X.shape[1] != self.n_features:
            raise ValueError("X must have {self.n_features} features")
        # ?
        return np.array([self._predict_single(x, self.root) for x in X])

    def best_split(self, X, y):
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

        OUTPUT:
            best_feat: (int) Index of the best feature to split on
        """
        if len(y) == 0 or X.shape[1] == 0:
            return None
        
        best_split_info = {
            "feature": None,
            "type": None,
            "threshold": None,
            "criterion_value": float("inf") if self.criterion == "gini" else
            -float("inf")
        }

        for feature in range(X.shape[1]):
            # Numerical split
            if self.feature_types[feature] == "numerical":
                split_info = self.split_numerical(X, y, feature)

            else:
                # Categorical split
                split_info = {
                    "type": "categorical",
                    "criterion_value": (
                        self.child_gini(X, y, feature) if self.criterion == \
                        "gini" else self.parent_entropy(y) -
                        self.child_entropy(X, y, feature)
                    )
                }

            # Compare criteria values and update if better
            if ((self.criterion == 'gini' and split_info['criterion_value'] < best_split_info['criterion_value']) or \
               (self.criterion == 'entropy' and split_info['criterion_value'] > best_split_info['criterion_value'])):
                best_split_info.update(split_info)
                best_split_info['feature'] = feature

        return best_split_info if best_split_info["feature"] is not None else None

    def plurality_value(self, y: np.ndarray, random_state: Optional[int] =
                        None) -> None:
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
        values, counts = np.unique(y, return_counts=True)

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
        X: np.ndarray, 
        y: np.ndarray, 
        parent_y: Optional[np.ndarray] = None, 
        current_depth: int = 0
    ) -> None:
        """
        Recursive function that grows the tree, returning the completed tree.
        --------------------------------------------------------
        INPUT:
            X: (np.ndarray) Feature matrix
            y: (np.ndarray) Target vector
            parent_y: (np.ndarray) Parent node's target vector (for empty nodes)
            current_depth: (int; initially zero)

        OUTPUT:
            node: (Node class)
        """
        # If first call, set parent_y to current y
        if parent_y is None:
            parent_y = y

        # Establish node
        node = Node(data={"X": X, "y": y})

        # If examples empty, return PLURALITY_VALUE(parent_examples)
        if len(y) == 0 or len(y) < self.min_num_samples:
            # Added min_num_samples check
            node.predicted_class = self.plurality_value(parent_y)
            node.is_leaf = True
            return node 

        # IF all examples have same classification
        if len(np.unique(y)) == 1:
            node.predicted_class = y[0]
            node.is_leaf = True
            return node

        # If max depth reached or no more features
        if X.shape[1] == 0 or (self.max_depth is not None and current_depth >=
                               self.max_depth):
            # Get most common
            node.predicted_class = self.plurality_value(y)
            node.is_leaf = True
            return node

        # Find best split
        split_info = self.best_split(X, y)
        if split_info is None:
            node.predicted_class = self.plurality_value(y)
            node.is_leaf = True
            return node

        feature = split_info["feature"]
        node.split_on = feature

        # Numerical
        if split_info["type"] == "numerical":
            # threshold
            threshold = split_info["threshold"]
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            node.children = {
                "left": self.learn_decision_tree(
                    X[left_mask],
                    y[left_mask],
                    y,
                    current_depth+1
                ),
                "right": self.learn_decision_tree(
                    X[right_mask],
                    y[right_mask],
                    y,
                    current_depth+1
                )
            }
            node.threshold = threshold

        else:
            # Categorical
            for value in np.unique(X[:, feature]):
                mask = X[:, feature] == value
                node.children[value] = self.learn_decision_tree(
                    X[mask],
                    y[mask],
                    current_depth+1
                )
        
        return node
        
    def traverse_tree(self, x: np.ndarray, tree):
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

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for samples in X
        -------------------------------------------------
        INPUT:
            X: (np.ndarray) Training data

        OUTPUT:
            probabiities: (np.ndarray) Array of probabilities
        """
        predictions = self.predict(X)
        probabilities = np.zeros((len(X), len(self.classes_)))

        for i, prediction in enumerate(predictions):
            class_idx = np.where(self.classes_ == predictions)[0][0]
            probabilities[i, class_idx] = 1.0

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
        self._check_input(X, y)

        # GEt data
        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)

        # Determine feature types
        self.feature_types = [self._get_feature_type(X, i) for i in
                              range(X.shape[1])]

        # Build tree
        self.root = self.learn_decision_tree(X, y)
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
        from sklearn.tree import DecisionTreeClassifier

        # Create sklearn tree with same parameters
        sklearrn_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion = self.criterion,
            min_samples_split=self.min_num_samples
        )

        # Fit both trees
        sklearn_tree.fit(X, y)
        self.fit(X, y)

        # Get prediction
        our_predictions = self.predict(test_X)
        sklearn_predictions = sklearn_tree.predict(test_X)

        # Compare predictions
        accuracy_match = np.mean(our_predictions == sklearn_predictions)
        return our_predictions, sklearn_predictions, accuracy_match



# =======================================================
"""EXAMPLE USAGE"""
# Load and encode data for exam_results.csv dataset
data = pd.read_csv('exam_results.csv')
mappings = {
    'Exam Result': {'Pass': 1, 'Fail': 0},
    'Other online courses': {'Y': 1, 'N': 0},
    'Student background': {'Maths': 0, 'CS': 1, 'Other': 2},
    'Working Status': {'W': 1, 'NW': 0}
}
# Encode categorical variables using the mappings
for column, mapping in mappings.items():
    data[column] = data[column].map(mapping)

# Split into features (X) and target (y)
X = data.drop('Exam Result', axis=1).values
y = data['Exam Result'].values
# =======================================================
# Test the implementation
dt = DecisionTree(max_depth=3, criterion="gini")
dt.fit(X, y)
predictions = dt.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")

# Validate against sklearn
#sklearn_comparison = dt.validate_using_sklearn(X, y, X)
#print(f"Sklearn accuracy match: {sklearn_comparison[2]:.2f}")
