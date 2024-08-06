#!/usr/bin/env python
"""
------------------------------------------------------------------------------------------------------------------------------
Parker, Cory 
AI Fundamentals graduate course at RIT
7/10/24 -- Actual start of project

The Help:
    - https://medium.com/@cristianleo120/master-decision-trees-and-building-them-from-scratch-in-python-af173dafb836
------------------------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
from math import log2

def user_attribute_adjustments(dataframe):
    """
    Asks user if they want to remove a column from the dataframe before
    processing data.
    ------------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame)

    OUTPUT:
        (pd.DataFrame) Without specified column
    """
    print("Current DataFrame")
    print(dataframe)

    remove = input("Do you want to remove a column? (yes/no)")

    if remove == "yes":
        print(f"Available columns: {list(dataframe.columns)}")

        remove_column = input("Enter the column to be removed: ").strip()

        if remove_column in dataframe.columns:
            dataframe = dataframe.drop(columns=[remove_column])
            print(f"Column: {remove_column} has been removed.")

        else:
            print(f"Column {remove_column} doesn't exist in dataframe.")

    else:
        print("\nNo column removed.")

    return dataframe

def read_data(filename, target_attribute):
    """
    Reads data from csv file. 
    ----------------------------------------
    INPUT:
        filename: (str) Filename including path.
        target_attribute: (str)

    OUTPUT:
        (tuple)
    """
    # Reads csv file
    X = pd.read_csv(filename)

    # Ask user if there's a column to be removed
    X = user_attribute_adjustments(X)
    
    # Moves target attribute to end of dataframe if not there already
    if X.columns[-1] != target_attribute:
        target = X.pop(target_attribute)
        X[target_attribute] = target

    y = X[X.columns[-1]]
    X = X.drop(columns=[target_attribute])
    y_encoded = pd.Series(pd.factorize(y)[0], name=target_attribute)
    
    return X, y_encoded.astype(float)

def data_entropy(y):
    """
    Entropy based on class probabilities for the dataset.
    ----------------------------------------------------
    INPUT:
        y: (pd.Series or np.array) Classifications.

    OUTPUT:
        (float) Entropy.
    """
    total = len(y)
    probabilities = y.value_counts() / total # pd.Series

    return -sum(p * log2(p) for p in probabilities if p > 0)

def attribute_entropy(X, y, attribute, threshold=None):
    """
    Weighted entropy for attribute, automatically adjusting for
    continuous/categorical attribute values.
    --------------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute
        attribute: (str)

    OUTPUT:
        weighted_entropy: (float)
    """
    total = len(y)

    if pd.api.types.is_numeric_dtype(X[attribute]) and threshold is not None:
        # continuous variables
        for value in X[attribute].unique():
            # Masks
            left_mask = X[X[attribute] <= value]
            right_mask = ~left_mask
            
            # Subsets; we only need the corresponding target values
            left_y = y[left_mask]
            right_y = y[right_mask]

            # Weights
            left_weight = len(left_y) / total
            right_weight = len(right_y) / total

            # Entropies
            left_entropy = data_entropy(left_y)
            right_entropy = data_entropy(right_y)

            # Weighted entropy
            weighted_entropy = left_entropy * left_weight + (right_entropy *
            right_weight)

        else:
            # Inititalze weighted entropy
            weighted_entropy = 0

            # Categorical variables
            for value in X[attribute].unique():
                subset_y = y[X[attribute] == value]
                subset_total = len(susbet_y)
                weight = subset_y / subset_total

                weighted_entropy += data_entropy(subset_y) * weight

        return weighted_entropy


def information_gain(X, y, attribute, threshold=None):
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
    weighted_child_entropy = attribute_entropy(X, y, attribute, threshold)

    return parent_entropy - weighted_child_entropy

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
    CATEGORICAL ATTRIBUTE VALUES
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

    return weighted_gini

def find_best_split_ig(X, y):
    """
    Uses INFORMATION GAIN to minimize Cost Function.
    -------------------------------------------
    Find highest information gain of all attributes, automatically determining
    whether attribute values are continuous or categorical.
    Looping thru each attribute, calculating the weighted average entropy,
    subtracing each from the parent entropy, where the attribute with the
    hightest information gain is returned to find the ROOT NODE to split on.
    --------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute

    OUTPUT:
        best_attribute, best_threshold: (tuple: (str), (float))
    """ 
    best_attribute = None
    best_ig = -1.0
    best_threshold = None

    for attribute in X.columns:
        # Continuous attribute values
        if pd.api.types.is_numeric_dtype(X[attribute]):
            sorted_values = X[attribute].sort_values().unique()
            midpoints = [(sorted_values[i] + sorted_values[i+1]) / 2 for i in
                         range(len(sorted_values) - 1)]

            # Partition ? 
            for threshold in midpoints:
                left_mask = X[attribute] <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                left_entropy = data_entropy(left_y)
                right_entropy = data_entropy(right_y)

                left_weight = len(left_y) / len(y)
                right_weight = len(right_y) / len(y)

                weighted_entropy = left_weight * (left_entropy + right_weight *
                right_entropy)
                # Information gain
                ig = data_entropy(y) - weighted_entropy

                # Compare current information gain to highest one so far
                if ig > best_ig:
                    best_ig = ig
                    best_attribute = attribute
                    best_threshold = threshold
    
        # Categorical attribute values
        else:
            ig = information_gain(X, y)

            if ig > best_ig:
                best_ig = ig
                best_attribute = attribute
                best_threshold = None

    return best_attribute, best_threshold

def find_best_split_gini(X, y):
    """
    Uses GINI INDEX to minimize the cost function.
    -----------------------------------------------
    Finds lowest gini index for attributes, automatically determining whether
    attribute values are categorical or continuous.
    Loops thru each attribute, calculating the weighted gini index, summing the
    product of the weights and the attribute gini indices.
    -----------------------------------------------
    INPUT:
        X: (pd.DataFrame) Attribute data
        y: (pd.Series) Target attribute

    OUTPUT:
        best_attribute, best_threshold: (tuple: (str), (float))
    """
    # Initialize with large value since gini ~ 1/info_gain
    best_attribute = None
    best_gini = float("inf")
    best_threshold = None

    for attribute in X.columns:
        # Check attribute data types
        if pd.api.types.is_numeric_dtype(X[attribute]):
            # Continuous attribute values
            sorted_values = X[attribute].sort_values().unique()
            midpoints = [(sorted_values[i] + sorted_values[i+1]) / 2 for i in
                        range(len(sorted_values) - 1)]
            
            # Compare threshold values
            for threshold in midpoints:
                left_mask = X[attribute] <= threshold
                right_mask =  ~left_mask
                
                # Partiion
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                left_weight = len(left_y) / len(y)
                right_weight = len(right_y) / len(y)
                
                left_gini = data_gini_index(left_y)
                right_gini = data_gini_index(right_y)
    
                weighted_gini = left_weight * left_gini + (right_weight *
                right_gini)
                
                # Determine the best gini, attribute, and threshold
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_attribute = attribute
                    best_threshold = threshold   
        
        else:
            # Categorical attribute values
            gini = attribute_gini_index(X, y, attribute)

            if gini < best_gini:
                best_gini = gini
                best_attribute = attribute

    return best_attribute, best_threshold

def grow_tree(X, y, max_depth=None, min_num_samples=2, current_depth=0,
              func=find_best_split_ig):
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
        my_tree: (dict): (tree: (dict), pass/fail: (dict))
    """
    # ------------------
    # Stopping criteria
    # ------------------
    # Target classes all the same
    if len(y.unique()) == 1:
#        print(f"All labels are the same!\ndepth: {current_depth}")
        return y.unique()[0]

    # Check for minimum number of samples, returning the mode (most common) if
    # so, where "samples" refers to the rows of X.
    if len(X) < min_num_samples:
        print(f"X is less than minimum number of samples")
        return y.mode().iloc[0]

    # Exceeding the max depth, returning the mode if so
    if max_depth is not None and current_depth >= max_depth:
        print(f"current_depth :{current_depth}")
        return y.mode().iloc[0]

    best_feature, best_threshold = func(X, y)

    if best_feature is None:
        print(f"best featture: {best_feature}")
        return y.mode().iloc[0]

    # Root node
    best_feature, best_threshold = func(X, y)

    # Ensure no nonsense
    if best_feature is None:
        return y.mode().iloc[0]

    # Initialize root node
    my_tree = {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": None,
        "right": None
    }

    # Split data depending on data types for features
    if pd.api.types.is_numeric_dtype(X[best_feature]):
        left_mask = X[best_feature] <= best_threshold

    else:
        left_mask = X[best_feature] == best_threshold

    # Partitioning
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[~left_mask], y[~left_mask]

    # Recursively grow left and right subtrees
    my_tree["left"] = grow_tree(X_left, y_left, max_depth, min_num_samples,
                                current_depth+1, func)
    my_tree["right"] = grow_tree(X_right, y_right, max_depth, min_num_samples,
    current_depth+1, func)

    return my_tree



# Usage
X, y = read_data("data/merged.csv", "Diagnosis")
#X, y = read_data("data/exam_results.csv", "Exam Result")
breakpoint()

