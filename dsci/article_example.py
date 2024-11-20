#!/usr/bin/env python
"""
Predicts whether student will pass class based on whether the student works, is
taking other online coures, or has a background in Computer Science, Math, or
Other.
------------------------------------------------------------------------------------------------------------------------------
Source: https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c
------------------------------------------------------------------------------------------------------------------------------
For BINARY CLASSIFICATION, but can easily be adjusted for MULTICLASS
classification...
------------------------------------------------------------------------------------------------------------------------------
"""
from math import log2
import pandas as pd
import graphviz
import time

dframe = pd.read_csv("exam_results.csv")
df = dframe.drop(columns=["Resp srl no"])
y = df["Exam Result"]
X = df.drop(columns=["Exam Result"])
df = df.reindex(columns=["Other online courses", "Student background", "Working Status", "Exam Result"])

def data_entropy(y):
    """
    INPUT:
        y: (pd.Series)

    OUTPUT:
        entropy: (float)
    """
    total = len(y)
    probabilities = y.value_counts() / total # Outputs pd.Series of probs

    return -sum([p * log2(p) for p in probabilities if p > 0])

def attribute_entropy(X, y, attribute, indent=""):
    """
    Weighted entropy for attribute.
    --------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute
        attribute: (str)

    OUTPUT:
        weighted_attribute_entropy: (float)
    """
    total = len(y)
    # Unique values only!
    attribute_values = X[attribute].unique()
    weighted_entropy = 0

    print(f"{indent} Calculating Entropy for attribute '{attribute}:'")

    for value in attribute_values:
        subset_y = y[X[attribute] == value]
        subset_total = len(subset_y)
        weight = subset_total / total
        subset_entropy = data_entropy(subset_y)

        # Pass/Fail counts
        pass_count = sum(subset_y == "Pass")
        fail_count = sum(subset_y == "Fail")
        weighted_entropy += weight * subset_entropy

        # Output info
        print(f"{indent}  {attribute} = {value}:")
        print(f"{indent}    Count: {len(subset_y)} (Pass: {pass_count}, Fail: {fail_count})")
        print(f"{indent}    Entropy: {weighted_entropy:.4f}")
        print(f"{indent}    Weight: {weight:.4f}")
    print(f"{indent}  Weighted Entropy for '{attribute}': {weighted_entropy:.4f}\n")

    return weighted_entropy

def information_gain(X, y, attribute, indent=""):
    """
    Entropy reduction
    --------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute
        attribute: (str) Column

    OUTPUT:
        info_gain: (float)
    """
    parent_entropy = data_entropy(y)
    weighted_child_entropy = attribute_entropy(X, y, attribute)
    info_gain = parent_entropy - weighted_child_entropy
    print(f"{indent}Information Gain by splitting on '{attribute}': {info_gain:.4f}\n")

    return info_gain

def data_gini_index(y):
    """
    For determining the ROOT NODE.
    ------------------------------------------------------
    Gini Index or Impurity measures the probability for a random instance being misclassified when chosen randomly. The lower the Gini Index, the better the lower the likelihood of misclassification.
    -------------------------------------------------------
    Gini Index = 1 - \sum_{i=1}^{j} p(j)^2, Where j represents the no. of classes in the target variable — Pass and Fail in our example
    P(i) represents the ratio of Pass/Total no. of observations in node.
     It has a maximum value of .5. If Gini Index is .5, it indicates a random assignment of classes.
    -------------------------------------------------------
    INPUT:
        y: (pd.Series) Target attribute

    OUTPUT:
        gini index for data: (float)
    """
    total = len(y)
    probabilities = y.value_counts() / total

    return 1 - sum(probabilities**2)

def attribute_gini_index(X, y, attribute):
    """
    Calculates the weighted Gini index for a given attribute.
    ------------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute
        attribute: (str) Column

    OUTPUT:
        weighted_gini: (float)
    """
    total = len(y)
    attribute_values = X[attribute].unique()
    weighted_gini = 0

    for value in attribute_values:
        subset_y = y[X[attribute] == value]
        subset_total = len(subset_y)
        weight = subset_total / total
        subset_gini = data_gini_index(subset_y)
        weighted_gini += weight * subset_gini
        # Output info
        print(f"{indent}  {attribute} = {value}:")
        print(f"{indent}    Count: {len(subset_y)} (Pass: {pass_count}, Fail: {fail_count})")
        print(f"{indent}    Gini: {gini:.4f}")
        print(f"{indent}    Weight: {weight:.4f}")
    print(f"{indent}  Weighted Gini for '{attribute}': {weighted_gini:.4f}\n")

    return weighted_gini

def find_best_split_ig(X, y):
    """
    Uses INFORMATION GAIN.
    -------------------------------------------
    Find highest information gain of all attributes.
    Looping thru each attribute, calculating the weighted average entropy,
    subtracing each from the parent entropy, where the attribute with the
    hightest information gain is returned to find the ROOT NODE to split on.
    --------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute

    OUTPUT:
        node: (str) Attribute for split 
    """ 
    best_attribute = None
    best_ig = -1.0

    for attribute in X.columns:
        ig = information_gain(X, y, attribute)
#        print(f"ig: {ig}\nattribute: {attribute}")

        if ig > best_ig:
            best_ig = ig
            best_attribute = attribute

    print(f"Best attribute: {best_attribute}")
    print(f"Best information gain: {best_ig}")

    return best_attribute

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
    best_gini = float("-inf")

    for attribute in X.columns:
        gini = attribute_gini_index(X, y, attribute)

        if gini < best_gini:
            best_gini = gini
            best_attribute = attribute

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
    # ------------------
    # Stopping criteria
    # ------------------
    # Target classes all the same
    if len(y.unique()) == 1:
        print(f"{indent}All samples have the same label: {y.unique()[0]}")

        # Return counts for leaf node
        return {"Pass": pass_count, "Fail": fail_count}

    # Check for minimum number of samples, returning the mode (most common) if
    # so, where "samples" refers to the rows of X.
    if len(X) < min_num_samples:
        majority_class = y.mode().iloc[0]
        print(f"{indent}No further attributes or max depth reached. Predicting majority class: {majority_class}")

        return {"Pass": pass_count, "Fail": fail_count}

    # Exceeding the max depth, returning the mode if so
    if max_depth is not None and current_depth >= max_depth:
        majority_class = y.mode().iloc[0]
        print(f"{indent}No further attributes or max depth reached. Predicting majority class: {majority_class}")
    
        return {"Pass": pass_count, "Fail": fail_count}

    best_feature = func(X, y)

    # No best information gain, return most common item
    if best_feature is None:
        return {"Pass": pass_count, "Fail": fail_count}

    # Calculate information gain for each attribute
    info_gains = {}
    for attribute in X.columns:
        print(f"{indent} Evaluating attribute: '{attribute}':")
        ig = information_gain(X, y, attribute, indent + "   ")
        info_gains[attribute] = ig

    # Select the best attribute based on highest information gain
    best_attribute = max(info_gains, key=info_gains.get)
    best_info_gain = info_gains[best_attribute]
    print(f"""{indent} Best attribute to split on: '{best_attribute}' with
          information gain: {best_info_gain:.4f}""")
    print(f"{indent}---------------------------------------------------\n")
    # Initialize tree with ROOT NODE
    my_tree = {best_attribute: {}}

    # Split dataset and grow subtrees for each split
    for feature_value in X[best_feature].unique():
        print(f"{indent}Splitting '{best_attribute}' = '{feature_value}'")
        subset_X = X[X[best_feature] ==
                     feature_value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == feature_value]

        # When leaf node reached, attach Pass/Fail values, otherwise continue
        # branching
        if len(subset_X.columns) == 0 or len(subset_y.unique()) == 1:
            my_tree[best_attribute][feature_value] = {
                "Pass": sum(subset_y == "Pass"),
                "Fail": sum(subset_y == "Fail")
            }

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
    Removes the unneccessary Pass/Fail summary labels for the tree in order to
    correctly "process" the predictions.
    ------------------------------------------
    INPUT:
        tree: (dict)

    OUTPUT:
        cleaned_tree: (dict)
    """
    # Esnures tree is a dict
    if isinstance(tree, dict):
        # Removes the keys Pass/Fail for making predictions
        subtree = {k: v for k, v in tree.items() if k not in ["Pass", "Fail"]}
        
        # BAD case -- Do something different with this ... ?
        if not subtree:
            pass_count = tree.get("Pass", 0)
            fail_count = tree.get("Fail", 0)
            
            return "Pass" if pass_count > fail_count else "Fail"

        return {k: clean_tree(v) for k, v in subtree.items()}
    
    else:
        return tree

def predict(tree, test, indent=""):
    """
    Predictions with debugging output.
    --------------------------------
    INPUT:
        tree: (dict) Decision Tree model
        test: (dict) Test data
        indent: (str)

    OUTPUT:
        prediction: (?)
    """
    print(f"{indent} Current tree node: ", tree)

    # If leaf node, non-dictionary
    if not isinstance(tree, dict):
        print(f"{indent}Reached leaf node with prediction: ", tree)

        return tree
   
    # Pass/Fail node
    if set(tree.keys()) <= {"Pass", "Fail"}:
        prediction = "Pass" if tree["Pass"] > tree["Fail"] else "Fail"
        print(f"""{indent}Reached leaf node with count - Pass: {tree.get('Pass',
              0)}, Fail: {tree.get('Fail', 0)}""")
        print(f"{indent}Predicting: ", prediction)

        return prediction

    for attribute, subtree in tree.items():
        print(f"{indent}Testing attribute: {attribute}")
        next_attribute = test[attribute]
        print(f"{indent}Test case has value: {next_attribute}")
        
        if next_attribute in subtree:
            print(f"{indent}Following path: {attribute} = {next_attribute}")
            prediction = predict(subtree[next_attribute], test)

            return prediction
        
        else:
            print(f"""{indent}WARNING: Value {next_attribute} not found in
                  training data for {attribute}""")
            # Return the majority class or handle unseen values



# --------------------------------
# VISUALIZATION of Decision Tree
# --------------------------------
# ChatGPT -- Do yourself ?
#def visualize_tree(tree, node_name='root', graph=None):
#    """
#    Visualize the decision tree using graphviz.
#    ------------------------------------------------
#    INPUT:
#        tree: (dict) The decision tree
#        node_name: (str) The name of the current node (default is 'root')
#        graph: (graphviz.Digraph) The graphviz graph object (default is None)
#
#    OUTPUT:
#        graph: (graphviz.Digraph) The graphviz graph object with the tree added
#    """
#    if graph is None:
#        graph = graphviz.Digraph()
#
#    if not isinstance(tree, dict):
#        graph.node(node_name, label=str(tree), shape='ellipse')
#        return graph
#
#    for attribute, subtree in tree.items():
#        if attribute not in ["Pass", "Fail"]:
#            for value, subsubtree in subtree.items():
#                child_node_name = f'{node_name}_{value}'
#                graph.node(child_node_name, label=f'{value}')
#                graph.edge(node_name, child_node_name, label=attribute)
#                visualize_tree(subsubtree, node_name=child_node_name, graph=graph)
#        else:
#            graph.node(node_name, label=f'{tree["Pass"]}P, {tree["Fail"]}F', shape='box')
#
#    return graph

# Usage:
#graph = visualize_tree(cleaned_tree)
#graph.render('decision_tree', format='png', cleanup=True)


# -------------------------------------------------------------------------------------------------------------------------
#+++++++++++++
#EXAMPLE USAGE:
#+++++++++++++
def main():
    #tree = grow_tree(X, y, func=find_best_split_gini)
    tree = grow_tree(X, y)
    cleaned_tree = clean_tree(tree)
    #test_data = {"Other online courses": "Y", "Student background": "Maths", "Working Status": "NW"}
    test_data = {"Other online courses": "Y", "Student background": "Maths", "Working Status": "W"}
    prediction = predict(cleaned_tree, test_data)
    print(f"The predicted class is: {prediction}")

if __name__ == "__main__":
    main()


