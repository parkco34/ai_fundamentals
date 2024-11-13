#!/usr/bin/env python
import numpy as np


class Node:
    def __init__(self):
        pass


class DecisionTree:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini"
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None

    def impurity(self):
        pass

    def info_gain(self):
        pass

    def best_split(self):
        pass

    def create_tree(self, X, y, depth):
        pass

    def fit(self, X, y):
        pass

    def traverse_tree(self, X, node):
        pass

    def prediction(self, X):
        pass

    def get_params(self):
        pass


