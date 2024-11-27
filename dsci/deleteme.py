#!/usr/bin/env python
class DecisionTree(object):
    # ... [other methods] ...

    def fit(self, X, y):
        self.tree = self.learn_decision_tree(
            X,
            y,
            max_depth=self.max_depth,
            min_num_samples=self.min_num_samples,
            method=self.criterion
        )

    def predict(self, X):
        """
        Predict the class labels for the given test data.

        INPUT:
            X: (np.ndarray) Test data

        OUTPUT:
            y_pred: (np.ndarray) Predicted class labels
        """
        y_pred = [self.predict_single(sample, self.tree) for sample in X]
        return np.array(y_pred)

    def predict_single(self, sample, tree):
        """
        Recursively traverse the tree to predict the class label for a single sample.

        INPUT:
            sample: (np.ndarray) A single sample
            tree: (dict) The decision tree

        OUTPUT:
            label: Predicted class label
        """
        if 'class' in tree:
            return tree['class']
        else:
            feature = tree['feature']
            value = sample[feature]
            if value in tree['branches']:
                subtree = tree['branches'][value]
                return self.predict_single(sample, subtree)
            else:
                # Use the plurality class stored at this node
                return tree['plurality_class']

    def plurality_value(self, y, random_state=None):
        """
        Returns the most common output value among set of examples, breaking
        ties randomly.
        """
        if len(y) == 0:
            return None

        values, counts = np.unique(y, return_counts=True)
        max_count_indices = np.where(counts == counts.max())[0]

        if len(max_count_indices) == 1:
            return values[max_count_indices[0]]

        rng = np.random.default_rng(random_state)
        return rng.choice(values[max_count_indices])

    def learn_decision_tree(self, X, y, parent_y=None, max_depth=None, min_num_samples=2,
                            current_depth=0, method="gini"):
        """
        Recursive function that grows the tree, returning the completed tree.
        """
        # ... [existing code] ...

        # Find the plurality class of the current node
        plurality_class = self.plurality_value(y)

        # If stopping conditions are met, return a leaf node
        if len(y) == 0:
            return {"class": self.plurality_value(parent_y)}
        if len(np.unique(y)) == 1:
            return {"class": y[0]}
        if X.shape[1] == 0 or (max_depth is not None and current_depth >= max_depth):
            return {"class": plurality_class}

        # Find best attribute to split on
        best_feature = self.best_split(X, y, method=method)

        if best_feature is None:
            return {"class": plurality_class}

        # Create the tree node with plurality class
        tree = {
            "feature": best_feature,
            "branches": {},
            "plurality_class": plurality_class
        }

        # For each value of the best feature
        for value in np.unique(X[:, best_feature]):
            mask = X[:, best_feature] == value
            X_subset = np.delete(X[mask], best_feature, axis=1)
            y_subset = y[mask]

            subtree = self.learn_decision_tree(
                X_subset,
                y_subset,
                parent_y=y,
                max_depth=max_depth,
                min_num_samples=min_num_samples,
                current_depth=current_depth + 1,
                method=method
            )

            tree["branches"][value] = subtree

        return tree

    # ... [other methods] ...


