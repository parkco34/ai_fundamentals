#!/usr/bin/env python

# Logging:
import logging
from graphviz import Digraph
from math import log2
import pandas as pd
import graphviz

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Assuming the data is already loaded into X and y
# For example:
# df = pd.read_csv("exam_results.csv")
# y = df["Exam Result"]
# X = df.drop(columns=["Exam Result"])

def find_best_split_gini(X, y):
    """
    Uses GINI INDEX.
    -----------------------------------------------
    Finds lowest gini index for attributes.
    Loops thru each attribute, calculating the weighted gini index, summing the
    product of the weights and the attribute gini indices.
    -----------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute

    OUTPUT:
        node: (str) Attribute for split
    """
    # Initialize with large value since gini ~ 1/info_gain
    best_attribute = None
    best_gini = float("inf")

    for attribute in X.columns:
        gini = attribute_gini_index(X, y, attribute)

        if gini < best_gini:
            best_gini = gini
            best_attribute = attribute

    # Logging
    logging.debug(f"Best Attribute: {best_attribute}")
    logging.debug(f"Best Gini index: {best_gini}")

    # Output info
    print(f"Best attribute: {best_attribute}")
    print(f"Best Gini index: {best_gini}")

    return best_attribute

def grow_tree(X, y, max_depth=None, min_num_samples=2, current_depth=0,
              func=find_best_split_ig, indent=""):
    """
    Recursive function that grows the tree, returning the completed tree.
    ------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute
        max_depth: (int; default: None)
        min_num_samples: (int; default: 2)
        current_depth: (int; default: 0)
        func: (function; default: find_best_split_ig) Function to find best
            split with; Information gain or Gini Index.

    OUTPUT:
        (dict): (tree: (dict), pass/fail: (dict))
    """
    total_samples = len(y)
    pass_count = sum(y == "Pass")
    fail_count = sum(y == "Fail")
    parent_entropy = data_entropy(y)

    print(f"{indent}==================== Depth {current_depth} ====================")
    print(f"{indent}Total samples: {total_samples} (Pass: {pass_count}, Fail: {fail_count})")
    print(f"{indent}Parent Entropy: {parent_entropy:.4f}\n")

    # Logging
    logging.debug(f"{indent}Depth {current_depth}, Total samples: {total_samples}")
    logging.debug(f"{indent}Pass: {pass_count}, Fail: {fail_count}")
    logging.debug(f"{indent}Parent Entropy: {parent_entropy:.4f}")

    # ------------------
    # Stopping criteria
    # ------------------
    # Target classes all the same
    if len(y.unique()) == 1:
        print(f"{indent}All samples have the same label: {y.unique()[0]}")

        # Logging
        logging.debug(f"{indent}All samples have the same label: {y.unique()[0]}")

        # Return counts for leaf node
        return {"Pass": pass_count, "Fail": fail_count}

    # Check for minimum number of samples, returning the mode (most common) if
    # so, where "samples" refers to the rows of X.
    if len(X) < min_num_samples:
        majority_class = y.mode().iloc[0]
        print(f"{indent}No further attributes or max depth reached. Predicting majority class: {majority_class}")

        # Logging
        logging.debug(f"{indent}Sample size {len(X)} less than min_num_samples {min_num_samples}")
        logging.debug(f"{indent}Predicting majority class: {majority_class}")

        return {"Pass": pass_count, "Fail": fail_count}

    # Exceeding the max depth, returning the mode if so
    if max_depth is not None and current_depth >= max_depth:
        majority_class = y.mode().iloc[0]
        print(f"{indent}No further attributes or max depth reached. Predicting majority class: {majority_class}")

        # Logging
        logging.debug(f"{indent}Current depth {current_depth} exceeds max depth {max_depth}")
        logging.debug(f"{indent}Predicting majority class: {majority_class}")

        return {"Pass": pass_count, "Fail": fail_count}

    best_attribute = func(X, y)

    # Logging
    logging.debug(f"{indent}Best attribute selected: {best_attribute}")

    # No best information gain, return most common item
    if best_attribute is None:
        return {"Pass": pass_count, "Fail": fail_count}

    # Optionally calculate and print the information gain for the best attribute
    best_info_gain = information_gain(X, y, best_attribute, indent)

    print(f"""{indent} Best attribute to split on: '{best_attribute}' with
          information gain: {best_info_gain:.4f}""")
    print(f"{indent}---------------------------------------------------\n")

    # Logging
    logging.debug(f"{indent}Best attribute to split on: '{best_attribute}' with information gain: {best_info_gain:.4f}")

    # Initialize tree with ROOT NODE
    my_tree = {best_attribute: {}}

    # Split dataset and grow subtrees for each split
    for feature_value in X[best_attribute].unique():
        print(f"{indent}Splitting '{best_attribute}' = '{feature_value}'")

        # Logging
        logging.debug(f"{indent}Splitting '{best_attribute}' = '{feature_value}'")

        subset_X = X[X[best_attribute] ==
                     feature_value].drop(columns=[best_attribute])
        subset_y = y[X[best_attribute] == feature_value]

        # When leaf node reached, attach Pass/Fail values, otherwise continue
        # branching
        if len(subset_X.columns) == 0 or len(subset_y.unique()) == 1:
            my_tree[best_attribute][feature_value] = {
                "Pass": sum(subset_y == "Pass"),
                "Fail": sum(subset_y == "Fail")
            }

            # Logging
            logging.debug(f"{indent}Leaf node created at '{best_attribute}' = '{feature_value}' with counts: Pass = {sum(subset_y == 'Pass')}, Fail = {sum(subset_y == 'Fail')}")
        else:
            my_tree[best_attribute][feature_value] = grow_tree(
                subset_X,
                subset_y,
                max_depth,
                min_num_samples,
                current_depth+1,
                func,
                indent + "  "
            )

    return my_tree

def clean_tree(tree):
    """
    Removes the unnecessary Pass/Fail summary labels for the tree in order to
    correctly "process" the predictions.
    ------------------------------------------
    INPUT:
        tree: (dict)

    OUTPUT:
        cleaned_tree: (dict)
    """
    # Ensures tree is a dict
    if isinstance(tree, dict):
        # Removes the keys Pass/Fail for making predictions
        subtree = {k: v for k, v in tree.items() if k not in ["Pass", "Fail"]}

        # Check if we have reached a leaf node
        if not subtree:
            pass_count = tree.get("Pass", 0)
            fail_count = tree.get("Fail", 0)

            # Logging
            logging.debug(f"Leaf node cleaned with counts: Pass = {pass_count}, Fail = {fail_count}")

            return "Pass" if pass_count > fail_count else "Fail"

        return {k: clean_tree(v) for k, v in subtree.items()}

    else:
        return tree

def get_majority_class(subtree):
    """
    Handles case when test data contains value not seen during training.
    --------------------------------------------------------
    INPUT:
        subtree: (dict)

    OUTPUT:
        decision: (str)
    """
    pass_wins, fail_wins = 0, 0

    # Count how many branches where Pass wins
    for branch_value, counts in subtree.items():
        if isinstance(counts, dict):
            if counts.get("Pass",  0) > counts.get("Fail", 0):
                pass_wins += 1
            else:
                fail_wins += 1

    decision = "Pass" if pass_wins > fail_wins else "Fail"

    # Logging
    logging.debug(f"Majority class determined: {decision}")

    return decision

def predict(tree, test, indent=""):
    """
    Predictions with debugging output.
    --------------------------------
    INPUT:
        tree: (dict) Decision Tree model
        test: (dict) Test data
        indent: (str)

    OUTPUT:
        prediction: (str)
    """
    print(f"{indent} Current tree node: ", tree)

    # Logging
    logging.debug(f"{indent}Current tree node: {tree}")

    # If leaf node, non-dictionary
    if not isinstance(tree, dict):
        print(f"{indent}Reached leaf node with prediction: ", tree)

        # Logging
        logging.debug(f"{indent}Reached leaf node with prediction: {tree}")

        return tree

    # Pass/Fail node
    if set(tree.keys()) <= {"Pass", "Fail"}:
        prediction = "Pass" if tree["Pass"] > tree["Fail"] else "Fail"
        print(f"""{indent}Reached leaf node with count - Pass: {tree.get('Pass', 0)}, Fail: {tree.get('Fail', 0)}""")
        print(f"{indent}Predicting: ", prediction)

        # Logging
        logging.debug(f"{indent}Reached leaf node with counts: Pass = {tree.get('Pass', 0)}, Fail = {tree.get('Fail', 0)}")
        logging.debug(f"{indent}Predicting: {prediction}")

        return prediction

    for attribute, subtree in tree.items():
        print(f"{indent}Testing attribute: {attribute}")
        next_attribute = test[attribute]
        print(f"{indent}Test case has value: {next_attribute}")

        # Logging
        logging.debug(f"{indent}Testing attribute: {attribute}")
        logging.debug(f"{indent}Test case has value: {next_attribute}")

        if next_attribute in subtree:
            print(f"{indent}Following path: {attribute} = {next_attribute}")

            # Logging
            logging.debug(f"{indent}Following path: {attribute} = {next_attribute}")

            prediction = predict(subtree[next_attribute], test, indent)
            return prediction

        else:
            print(f"""{indent}WARNING: Value {next_attribute} not found in training data for {attribute}""")

            # Logging
            logging.debug(f"{indent}WARNING: Value {next_attribute} not found in training data for {attribute}")

            # Return the majority class or handle unseen values
            majority = get_majority_class(subtree)
            return majority

def validate_using_sklearn(X, y, our_tree, test_data, max_depth=None):
    """
    Validates the decision tree implementation against
    scikit-learn's implementation.
    - Ensures both implementations use entropy as splitting criterion.
    - Uses same max_depth
    - Assumes test_data keys match X.columns
    -----------------------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Training data
        y: (pd.Series) Target labels
        our_tree: (dict) Decision tree
        test_data: (dict) Test instances
        max_depth: (int; optional, default=None) Max depth of tree

    OUTPUT:
        (tuple)
        our_prediction: (str)
        sklearn_prediction: (str)
        predictions_match: (bool) Whether predictions match
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        raise ImportError(
            "scikit-learn is required for validation. "
            "Install it using: pip install scikit-learn"
        )

    # Validate test_data keys match training data columns
    if not set(test_data.keys()) == set(X.columns):
        raise ValueError(
            f"Test data features {set(test_data.keys())} "
            f"don't match training features {set(X.columns)}"
        )

    # Create and train sklearn tree with matching parameters
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        random_state=73
    )
    clf.fit(X, y)

    # Get predictions from both implementations
    our_pred = predict(our_tree, test_data)

    # Convert test_data to format sklearn expects
    test_df = pd.DataFrame([test_data])

    sklearn_pred = clf.predict(test_df)[0]

    # Compare and print results
    predictions_match = our_pred == sklearn_pred
    print(f"Our prediction: {our_pred}")
    print(f"sklearn prediction: {sklearn_pred}")
    print(f"Predictions match: {predictions_match}")

    # Logging
    logging.debug(f"Our prediction: {our_pred}")
    logging.debug(f"sklearn prediction: {sklearn_pred}")
    logging.debug(f"Predictions match: {predictions_match}")

    return our_pred, sklearn_pred, predictions_match

def visualize_tree(tree, parent_name=None, edge_label=None, graph=None):
    if graph is None:
        graph = Digraph(format='png')
        graph.attr('node', shape='rectangle')

    if not isinstance(tree, dict):
        # Leaf node
        node_name = f"Leaf_{id(tree)}"
        graph.node(node_name, label=f"Predict: {tree}")

        if parent_name:
            graph.edge(parent_name, node_name, label=edge_label)

        # Logging
        logging.debug(f"Added leaf node: {node_name} with prediction: {tree}")

    else:
        for attribute, branches in tree.items():
            node_name = f"Node_{id(attribute)}"
            graph.node(node_name, label=attribute)

            if parent_name:
                graph.edge(parent_name, node_name, label=edge_label)

            # Logging
            logging.debug(f"Added decision node: {node_name} with attribute: {attribute}")

            for branch_value, subtree in branches.items():
                visualize_tree(subtree, parent_name=node_name, edge_label=str(branch_value), graph=graph)

    return graph

#+++++++++++++
#EXAMPLE USAGE:
#+++++++++++++
def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    # Load your data here
    # For example:
    # df = pd.read_csv("exam_results.csv")
    # y = df["Exam Result"]
    # X = df.drop(columns=["Exam Result"])

    # Build the tree
    tree = grow_tree(X, y)
    cleaned_tree = clean_tree(tree)

    # Test data
    test_data = {"Other online courses": "Y", "Student background": "Maths", "Working Status": "W"}

    # Prediction
    prediction = predict(cleaned_tree, test_data)
    print(f"The predicted class is: {prediction}")

    # Validate against scikit-learn
    our_pred, sklearn_pred, match = validate_using_sklearn(X, y, cleaned_tree, test_data)
    print(f"Validation result: {'Match' if match else 'Mismatch'}")

    # Visualize
    tree_graph = visualize_tree(cleaned_tree)
    tree_graph.render('article_example')

if __name__ == "__main__":
    main()


# ================================================================================
def main():
    # Build and clean the tree
    tree = grow_tree(X, y)
    cleaned_tree = clean_tree(tree)

    # Make a prediction
    test_data = {"Other online courses": "Y", "Student background": "Maths", "Working Status": "W"}
    prediction = predict(cleaned_tree, test_data)
    print(f"The predicted class is: {prediction}")

    # Validate against scikit-learn
    our_pred, sklearn_pred, match = validate_using_sklearn(X, y, cleaned_tree, test_data)
    print(f"Validation result: {'Match' if match else 'Mismatch'}")

    # Visualize the tree
    tree_graph = visualize_tree(cleaned_tree)
    tree_graph.render('article_example')

try:
    dframe = pd.read_csv("exam_results.csv")
except FileNotFoundError:
    print("Error: The file 'exam_results.csv' was not found.")
#    return

required_features = set(X.columns)
if not required_features.issubset(test_data.keys()):
    missing_features = required_features - set(test_data.keys())
    print(f"Error: Missing features in test data: {missing_features}")
#    return


class DecisionTreeClassifierFromScratch:
    def __init__(self, max_depth=None, min_num_samples=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_num_samples = min_num_samples
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        if self.criterion == 'entropy':
            self.tree = grow_tree(X, y, self.max_depth, self.min_num_samples, func=find_best_split_ig)
        elif self.criterion == 'gini':
            self.tree = grow_tree(X, y, self.max_depth, self.min_num_samples, func=find_best_split_gini)
        self.tree = clean_tree(self.tree)

    def predict(self, test_data):
        return predict(self.tree, test_data)

    def visualize(self, filename='decision_tree'):
        tree_graph = visualize_tree(self.tree)
        tree_graph.render(filename)

