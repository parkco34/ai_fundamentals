#!/usr/bin/env python
def best_split(self, X, y, method="gini"):
    """
    Finds the best feature to split on using either Gini impurity or Information Gain.
    ------------------------------------------------------
    INPUT:
        X: (np.ndarray) Training data
        y: (np.ndarray) Test data
        method: (str) "gini" or "entropy" for information gain

    OUTPUT:
        best_feat: Best feature for split
    """
    # Initialize best feature and criterion value
    best_feat = None

    if method == "gini":
        # For Gini, we want to minimize
        best_criterion = float('inf')

        for feat in np.unique(X.columns):
            gini = self.child_gini(X, y, feat)

            # Update if we find a smaller Gini index
            if gini < best_criterion:
                best_criterion = gini
                best_feat = feat

    else:  # entropy/information gain
        # For information gain, we want to maximize
        best_criterion = -float('inf')

        for feat in np.unique(X.columns):
            # Calculate information gain
            parent_ent = self.parent_entropy(y)
            child_ent = self.child_entropy(X, y, feat)
            info_gain = parent_ent - child_ent

            # Update if we find larger information gain
            if info_gain > best_criterion:
                best_criterion = info_gain
                best_feat = feat

    # Logging
    logging.debug(f"Best Feature: {best_feat}")
    logging.debug(f"Best {'Gini' if method == 'gini' else 'Information Gain'}: {best_criterion}")

    return best_feat

def learn_decision_tree(self, X, y, parent_y=None, max_depth=None, min_num_samples=2,
                       current_depth=0, method="gini"):
    """
    Recursive function that grows the tree, returning the completed tree.
    --------------------------------------------------------
    INPUT:
        X: (np.ndarray) Feature matrix
        y: (np.ndarray) Target vector
        parent_y: (np.ndarray) Parent node's target vector (for empty nodes)
        max_depth: (int; default: None)
        min_num_samples: (int; default: 2)
        current_depth: (int; default: 0)
        method: (str) Split criterion ("gini" or "entropy")

    OUTPUT:
        tree: (dict) A dictionary representing the decision tree structure
    """
    # First call setup
    if parent_y is None:
        parent_y = y

    # If examples empty, return PLURALITY_VALUE(parent_examples)
    if len(y) == 0:
        return {"class": self.plurality_value(parent_y)}
    
    # If all examples have the same classification
    if len(np.unique(y)) == 1:
        return {"class": y[0]}
    
    # If attributes is empty or max depth reached
    if X.shape[1] == 0 or (max_depth is not None and current_depth >= max_depth):
        return {"class": self.plurality_value(y)}
    
    # Find best attribute using existing best_split method
    best_feature = self.best_split(X, y, method=method)
    
    if best_feature is None:
        return {"class": self.plurality_value(y)}
    
    # Create tree structure
    tree = {
        "feature": best_feature,
        "branches": {}
    }
    
    # For each value of the best feature
    for value in np.unique(X[:, best_feature]):
        # Create mask for this feature value
        mask = X[:, best_feature] == value
        
        # Get subset of data excluding the used feature
        X_subset = np.delete(X[mask], best_feature, axis=1)
        y_subset = y[mask]
        
        # Recursive call
        subtree = self.learn_decision_tree(
            X_subset,
            y_subset,
            parent_y=y,
            max_depth=max_depth,
            min_num_samples=min_num_samples,
            current_depth=current_depth + 1,
            method=method
        )
        
        # Add branch to tree
        tree["branches"][value] = subtree
    
    return tree
