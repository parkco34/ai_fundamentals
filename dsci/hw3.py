#!/usr/bin/env python
from textwrap import dedent
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# initiai CONSTANTS
RANDOM_STATE = 73
# Feature data used to train 80% since test_size=0.2
TEST_SIZE = 0.2

# Set random seed
np.random.seed(RANDOM_STATE)

# TASK 1: =======================================================
# Load dataset
# X: Flower measurements -> sepal length, sepal width, petal length, petal width for
# 120 flowers
# y: Species of flower, 0, 1, or 2
iris = load_iris()
X, y = iris.data, iris.target


# Hyperparameter grid setup +++++++++++++++++++++++++++++++++++++++
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# Define parameter dist. for Random Search
# "max_depth": Determines how deep the tree goes, with None meaning no limit.
# The smaller the value, the simpler the tree, less likely to overfit
# Larger values risk overfitting.
# "min_sample_split" -> Number of sample to split node w/ higher values meaning 
# "min_sample_leaf" -> Samples required at each leaf node w/ higher values for
# better balanced trees
# Criterion -> Quality of split, where Gini measures probability of incorrect
# classification and Entropy measuress info gain
params = {
    "max_depth": range(1, 21),
    # Min samples needed to split 2-20
    "min_samples_split": range(2, 21), 
    # leaf node: (1-10)
    "min_samples_leaf": range(1, 11),
    "criterion": ["gini", "entropy"]
}

# Random Search
def perform_random_search(X_train, y_train, params):
    """
    Implements RandomizedSearchCV to find optimal parameters for a Deciosion
    Tree Classifier, which is more efficient than GridSearchCV since samples
    random combinations rather than trying every possible combination.
    ------------------------------------------------------------
    INPUT:
        X_train: (np.ndarray) Training data
        y_train: (np.ndarray) Training labels
        params: (dict) Hyperparameters

    OUTPUT: 
    """
    # Initializae base classifier
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    
    # Random Search
    random_search = RandomizedSearchCV(
        estimator=dt,
        param_distributions=params,
        n_iter=100, # For good coverage ?
        cv=5, # 5-fold cross-validation
        scoring="accuracy",
        n_jobs=-1, # Use all available cores
        random_state=RANDOM_STATE,
        verbose=1 # To monitor  progress
    )

    random_search.fit(X_train, y_train)

    return random_search

def analyze_best_model(X_train, y_train, params):
    """
    Analyzes best model, outputting best parameters, cross-validatoin score,
    and model.
    --------------------------------------------------------------
    INPUT:
        X_train: (np.ndarray) Training data
        y_train: (np.ndarray) Training labels

    OUTPUT:
        best_model: 
    """
    random_search = perform_random_search(X_train, y_train, params)
    print("\nModel Analysis Results")
    print("_"*50)
    #Access parameters
    print(f"Best parameters: {random_search.best_params_}")
    # Best score
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")

    # Get best model
    best_model = random_search.best_estimator_

    # Training score
    train_score = best_model.score(X_train, y_train)
    print(f"Training accuracy: {train_score:.4f}")

    # Feature importances
    feature_importance = pd.DataFrame({
        "feature": iris.feature_names,
        "importance": best_model.feature_importances_
    })
    print(f"""\nFeature Importances:
          {feature_importance.sort_values('importance', ascending=False)}""")

    return best_model

def error_analysis(best_model, X_test, y_test):
    """
    Takes model, test data, and true labels to make predictions and identify
    misclassified samples
    --------------------------------------------------------------
    INPUT:
        best_model: (Fitted DecisionTreeClassifier)
        X_test: (np.ndarray) Test features
        y_test: (np.ndarray) True test labels

    OUTPUT:
        misclassified_indices: (np.ndarray) Indices of misclassified samples
    """
    # Predict!
    y_prediction = best_model.predict(X_test)

    # Find misclassified instances
    misclassified_mask = y_test != y_prediction
    misclassified_indices = np.where(misclassified_mask)[0]

    # Analysis time!
    print("\nError Analysis Results:")
    print("-"*50)
    print(f"Number of misclassified instances: {len(misclassified_indices)}")
    print(f"Test set accuracy: {accuracy_score(y_test, y_prediction):.4f}")

    if len(misclassified_indices) > 0:
        print("\nMisclassified Instances Details:")
        for i in misclassified_indices:
            print(f"\nIndex: {i}")
            print(f"True class: {iris.target_names[y_test[i]]}")
            print(f"Predicted class: {iris.target_names[y_prediction[i]]}")

            print("Features:")
            for feature_name, value in zip(iris.feature_names, X_test[i]):
                print(f"{feature_name}: {value:.2f}")


    # User input
    user_input_for_visual = input(
dedent("""
\nWould you like to see a visualization? 
('y'for yes, 'n' for no)\n
""")
                            )
    # Input validation
    while (user_input_for_visual != "y" and user_input_for_visual != "n"):
        user_input_for_visual = input("\nInput not valid.  Please try again.")
    
    # Determine whether or not to produce a visualization
    if user_input_for_visual == "y":
        visualization(X_test, y_test, y_prediction, misclassified_indices)

    return misclassified_indices

def visualization(X_test, y_test, y_pred, misclassified_indices):
    """
    Creates visualziations of the results anf misclassification.
    ---------------------------------------------------------
    INPUT:
        X_test: (np.ndarray) Test data
        y_test: (np.ndarray) Test labels
        misclassified_indices: (np.ndarray) Indices of misclassified samples

    OUTPUT:
        None
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Plot misclassified instances
    if len(misclassified_indices) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis",
                    alpha=0.6)
        plt.scatter(X_test[misclassified_indices, 0],
                    X_test[misclassified_indices, 1], c="red", marker="x",
                    s=200, label="Misclassified")
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1])
        plt.legend()
        plt.title("Misclassified Instances")
        plt.show()

def main():
    best_model = analyze_best_model(X_train, y_train, params)
    misclassified_indices = error_analysis(best_model, X_test, y_test)
    
if __name__ == "__main__":
    main()
