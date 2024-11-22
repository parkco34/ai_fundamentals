#!/usr/bin/env python
import logging

logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed output

def information_gain(X, y, attribute, indent=""):
    parent_entropy = data_entropy(y)
    weighted_child_entropy = attribute_entropy(X, y, attribute)
    info_gain = parent_entropy - weighted_child_entropy
    logging.debug(f"{indent}Information Gain by splitting on '{attribute}': {info_gain:.4f}\n")
    return info_gain

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

