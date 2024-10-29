#!/usr/bin/env python
"""
Part 1: Decision Trees (DT)- 4pts. each task
To complete the following tasks on DTs, you can use Python and libraries such as scikit-learn to
implement your code.
Task 1: Hyperparameter Tuning (Random Search)
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a DT classiWier.
4. Perform a Random Search to Wind the best hyperparameters for the DT classiWier. Search for
hyperparameters like max depth, min samples split, min samples leaf, and criterion. HInt: Use the
RandomizedSearchCV function from scikit-learn.
5. Print the best hyperparameters and the model’s accuracy with these hyperparameters.
Task 2: Error Analysis
1. After training the DT model with the best hyperparameters from Task 1, use this model to make
predictions on the test data.
2. Identify and print the indices of misclassiWied instances (where the true class is not equal to the
predicted class).
Task 3: Confusion Matrix
1. Calculate the confusion matrix for the model’s predictions on the test data.
2. Print the confusion matrix values (True Positives, True Negatives, False Positives, False
Negatives).
Note: The following Tasks 4 and 5 were not taught extensively in class for DTs. However, the
concepts were covered in liner regression, so I’d like you to give these a try w.r.t. DTs.
Task 4: Regression with DTs
1. Load a dataset suitable for regression (e.g., the Boston housing dataset from scikit-learn).
2. Split the dataset into training and testing sets.
3. Implement a DT regression model.
4. Train the model on the training data.
5. Calculate and print the mean squared error (MSE) on the testing data to assess the model’s
performance.
Task 5: Metrics Comparison
1. Compare the performance of the DT classiWier from Task 1 and the DT regression model from
Task 4.
2. Calculate and print relevant evaluation metrics for the classiWier (e.g., accuracy, precision, recall,
F1-score) and the regression model (e.g., MSE).
3. Discuss the results, including which model performed better and why.
"""
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error
)
import seaborn as sns
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
def load_split_data():
    """
    Loads both classification (iris) and regression (california) datasets and
    splits them.
    """
    # Classification data
    iris = load_iris()
    X_cls, y_cls = iris.data, iris.target

    # Hyperparameter grid setup +++++++++++++++++++++++++++++++++++++++
    # Split data
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls,
                                                                        y_cls, test_size=TEST_SIZE)

    # Regression data
    california = fetch_california_housing()
    X_reg, y_reg = california.data, california.target
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg,
                                                                       y_reg,
                                                                        test_size=TEST_SIZE)

    return (X_cls_train, X_cls_test, y_cls_train, y_cls_test,
            X_reg_train, X_reg_test, y_reg_train, y_reg_test,
            iris, california)

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

def analyze_classification_results(y_true, y_pred, class_names):
    """
    Performs classification analysis including Confusion Matrix.
    ------------------------------------------------------
    INPUT:
        y_true: (np.ndarray) Training data
        y_train: (np.ndarray) Training labels
        params: (dict) Hyperparamters

    OUTPUT:
        best_model: ()
    """
    # Confusion matrix
    cm =  confusion_matrix(y_true, y_pred)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nClassification Metrics")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print("-"*50)
    print("True Positives, False Positives, False Negatives for each class:")
    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        print(f"\nClass {class_names[i]}:")
        print(f"True Positives: {tp}")
        print(f"False Negatives: {fn}")

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return accuracy, precision, recall, f1

def train_evaluate_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains and evaluates Decision Tree Regression model.
    --------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    # Initialize stutf
    dt_reg = DecisionTreeRegressor(random_state=RANDOM_STATE)
    dt_reg.fit(X_train, y_train)

    # Predict
    y_pred = dt_reg.predict(X_test)

    # MSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\nRegression Metrics:")
    print("="*50)
    print(f"Mean Sqaured Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # Feature Importance Analysis
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": dt_reg.feature_importances_
    })
    print("\nFeature Importances (Regression):")
    print(feature_importance.sort_values("importance", ascending=False))
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values (Regression)")
    plt.show()

    return mse, rmse

def compare_models(cls_metrics, reg_metrics):
    """
    Compares performance of classification and regression models.
    --------------------------------------------------------
    INPUT:
        cls_metrics: ()
        reg_metrics: ()

    OUTPUT:
        
    """
    print("\nModdel Comparison")
    print("="*50)
    print("Classification Model Peformance")
    print(f"- Accuracy: {cls_metrics[0]:.4f}")
    print(f"- Precision: {cls_metrics[1]:.4f}")
    print(f"- Recall: {cls_metrics[2]:.4f}")
    print(f"- F1 Score: {cls_metrics[3]:.4f}")

    print("\nRegressionn Model Performance")
    print(f"- MSE: {reg_metrics[0]:.4f}")
    print(f"- RMSE: {reg_metrics[1]:.4f}")

    print("-"*50)
    print("1. Classification Model:")
    print("""\cmodel's performance can be evaluated thru its accuracy and
          f1-score""")
    print("""\t-High precision/recall indicates good balance betweenfalse
          positives and negatives""")
    print("\n\n2. Regression Model:")
    print("\t- RMSE provides error metric in same units as target variable")
    print("\t- Lower RMSE indicates better model performance")

def analyze_best_model(X_train, y_train, params, iris):
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

def error_analysis(best_model, X_test, y_test, iris):
    """
    Takes model, test data, and true labels to make predictions and identify
    misclassified samples
    --------------------------------------------------------------
    INPUT:
        best_model: (Fitted DecisionTreeClassifier)
        X_test: (np.ndarray) Test features
        y_test: (np.ndarray) True test labels
        iris: () Iris dataset object

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

def visualization(X_test, y_test, y_pred, misclassified_indices, iris):
    """
    Creates visualziations of the results anf misclassification.
    ---------------------------------------------------------
    INPUT:
        X_test: (np.ndarray) Test data
        y_test: (np.ndarray) Test labels
        misclassified_indices: (np.ndarray) Indices of misclassified samples
        iris: () Iris dataset object

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
    # Load split data
    (X_cls_train, X_cls_test, y_cls_train, y_cls_test, X_reg_train, X_reg_test,
    y_reg_train, y_reg_test, iris, california) = load_split_data()

    # Classification analysis
    best_model = analyze_best_model(X_cls_train, y_cls_train, params, iris)
    y_cls_prediction = best_model.predict(X_cls_test)
    cls_metrics = analyze_classification_results(y_cls_test, y_cls_prediction,
                                                iris.target_names)

    # Regression Analysis
    reg_metrics = train_evaluate_regression(X_reg_train, X_reg_test,
                                            y_reg_train, y_reg_test,
                                            california.feature_names)

    # Compare models
    compare_models(cls_metrics, reg_metrics)


if __name__ == "__main__":
    main()
