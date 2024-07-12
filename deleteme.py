#!/usr/bin/env python
"""
Study this and delete when you've got it!
"""
import numpy as np
import pandas as pd

def info_gain(X, y, attribute, threshold):
    mask = X[X.columns[attribute]] <= threshold
    left = entropy(y[mask])
    right = entropy(y[~mask])
    parent = entropy(y)
    w_left = sum(mask) / len(y)
    w_right = sum(~mask) / len(y)
    return parent - (w_left * left + w_right * right)

def best_split(X, y):
    num_attrs = X.shape[1]
    best_attr, best_threshold, max_gain = None, None, -1

    for feat in range(num_attrs):
        thresholds = np.unique(X[X.columns[feat]])
        for threshold in thresholds:
            # Threshold is np.float(type(float)) !!! ERROR !!!
            gain = info_gain(X, y, feat, threshold)

            if gain > max_gain:
                best_attr, best_threshold, max_gain = feat, threshold, gain

    return best_attr, float(best_threshold)

print(best_split(X, y_encoded))

