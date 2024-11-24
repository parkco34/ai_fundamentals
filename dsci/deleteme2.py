#!/usr/bin/env python
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


