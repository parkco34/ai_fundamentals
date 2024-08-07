#!/usr/bin/env python
import pandas as pd
from math import log2

# Core calculations
def data_entropy(y):
    probabilities = y.value_counts(normalize=True)
    return -sum(p * log2(p) for p in probabilities if p > 0)

def data_gini_index(y):
    probabilities = y.value_counts(normalize=True)
    return 1 - sum(p**2 for p in probabilities)

# Categorical attribute calculations
def categorical_attribute_entropy(X, y, attribute):
    return X[attribute].map(lambda val: data_entropy(y[X[attribute] == val])).mean()

def categorical_attribute_gini_index(X, y, attribute):
    return X[attribute].map(lambda val: data_gini_index(y[X[attribute] == val])).mean()

def categorical_information_gain(X, y, attribute):
    return data_entropy(y) - categorical_attribute_entropy(X, y, attribute)

def find_best_split_categorical(X, y, criterion='gini'):
    if criterion == 'gini':
        score_func = lambda attr: categorical_attribute_gini_index(X, y, attr)
        best_score = float('inf')
        compare = lambda score, best: score < best
    else:  # information gain
        score_func = lambda attr: categorical_information_gain(X, y, attr)
        best_score = -float('inf')
        compare = lambda score, best: score > best

    best_attribute = None
    for attribute in X.columns:
        score = score_func(attribute)
        if compare(score, best_score):
            best_score = score
            best_attribute = attribute

    return best_attribute, None, best_score

# Continuous attribute calculations
def continuous_split_score(X, y, attribute, threshold, criterion='gini'):
    left_mask = X[attribute] <= threshold
    right_mask = ~left_mask

    left_y, right_y = y[left_mask], y[right_mask]
    left_weight = len(left_y) / len(y)
    right_weight = 1 - left_weight

    if criterion == 'gini':
        return left_weight * data_gini_index(left_y) + right_weight * data_gini_index(right_y)
    else:  # information gain
        weighted_entropy = left_weight * data_entropy(left_y) + right_weight * data_entropy(right_y)
        return data_entropy(y) - weighted_entropy

def find_best_split_continuous(X, y, criterion='gini'):
    best_attribute = None
    best_threshold = None
    best_score = float('inf') if criterion == 'gini' else -float('inf')
    compare = (lambda x, y: x < y) if criterion == 'gini' else (lambda x, y: x > y)

    for attribute in X.columns:
        sorted_values = X[attribute].sort_values().unique()
        thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

        for threshold in thresholds:
            score = continuous_split_score(X, y, attribute, threshold, criterion)
            if compare(score, best_score):
                best_score = score
                best_attribute = attribute
                best_threshold = threshold

    return best_attribute, best_threshold, best_score

# General functions
def is_continuous(series):
    return series.dtype.kind in 'fiu' and len(series.unique()) > 10

def find_best_split(X, y, criterion='gini'):
    if all(is_continuous(X[attr]) for attr in X.columns):
        return find_best_split_continuous(X, y, criterion)
    else:
        return find_best_split_categorical(X, y, criterion)

def grow_tree(X, y, max_depth=None, min_samples=2, current_depth=0, criterion='gini'):
    if len(y.unique()) == 1 or len(X) < min_samples or (max_depth and current_depth >= max_depth):
        return y.mode().iloc[0]

    best_feature, threshold, _ = find_best_split(X, y, criterion)

    if best_feature is None:
        return y.mode().iloc[0]

    tree = {'feature': best_feature, 'threshold': threshold}

    if threshold is not None:  # Continuous attribute
        left_mask = X[best_feature] <= threshold
        right_mask = ~left_mask
        subtrees = [('left', left_mask), ('right', right_mask)]
    else:  # Categorical attribute
        subtrees = [(value, X[best_feature] == value) for value in X[best_feature].unique()]

    for value, mask in subtrees:
        if mask.sum() == 0:
            tree[value] = y.mode().iloc[0]
        else:
            tree[value] = grow_tree(
                X[mask].drop(columns=[best_feature]),
                y[mask],
                max_depth,
                min_samples,
                current_depth + 1,
                criterion
            )

    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature, threshold = tree['feature'], tree['threshold']

    if threshold is not None:  # Continuous attribute
        subtree_key = 'left' if sample[feature] <= threshold else 'right'
    else:  # Categorical attribute
        subtree_key = sample[feature] if sample[feature] in tree else max(tree, key=lambda x: x if isinstance(x, (int, float)) else 0)

    return predict(tree[subtree_key], sample)

def predict_all(tree, X):
    return X.apply(lambda x: predict(tree, x), axis=1)

# Example usage
if __name__ == "__main__":
    filename = "your_data.csv"
    target_attribute = "target"

    X, y = read_data(filename, target_attribute)

    for criterion in ['gini', 'information_gain']:
        tree = grow_tree(X, y, max_depth=5, criterion=criterion)
        print(f"\nDecision Tree ({criterion.capitalize()}):")
        print(tree)

        predictions = predict_all(tree, X)
        accuracy = (predictions == y).mean()
        print(f"Accuracy ({criterion.capitalize()}): {accuracy}")

