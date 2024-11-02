#!/usr/bin/env python
"""
------------------------
Part 1: 
Decision Trees (DT)
------------------------
Task 1: Hyperparameter Tuning (Random Search)
========================
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a DT classiWier.
4. Perform a Random Search to find the best hyperparameters for the DT classiWier. Search for
hyperparameters like max depth, min samples split, min samples leaf, and criterion. HInt: Use the
RandomizedSearchCV function from scikit-learn.
5. Print the best hyperparameters and the model’s accuracy with these hyperparameters.

Task 2: Error Analysis
========================
1. After training the DT model with the best hyperparameters from Task 1, use this model to make
predictions on the test data.
2. Identify and print the indices of misclassiWied instances (where the true class is not equal to the
predicted class).

Task 3: Confusion Matrix
========================
1. Calculate the confusion matrix for the model’s predictions on the test data.
2. Print the confusion matrix values (True Positives, True Negatives, False Positives, False
Negatives).
Note: The following Tasks 4 and 5 were not taught extensively in class for DTs. However, the
concepts were covered in liner regression, so I’d like you to give these a try w.r.t. DTs.

Task 4: Regression with DTs
========================
1. Load a dataset suitable for regression (e.g., the Boston housing dataset from scikit-learn).
2. Split the dataset into training and testing sets.
3. Implement a DT regression model.
4. Train the model on the training data.
5. Calculate and print the mean squared error (MSE) on the testing data to assess the model’s
performance.

Task 5: Metrics Comparison
========================
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



# Hyperparameters
def define_params(max_depth=range(1, 21), min_samples_split=range(2, 21),
                  min_samples_leaf=range(1, 11),
                  criterion=["gini", "entropy"]):
    """
    Defines parameter distribution for Random Search for the Decision Tree.
    ---------------------------------------------------------------
    "max_depth" -> Determine show deep the tree goes, with Noen meaning
    there's no limit.  The smaller the value, the simpler the tree; less
    likely to overfit.
        - Computational costs increases exponentially with increase in max
        depth.

    "min_samples_split"-> Number of samples to split node w/ higher values
    meaning, the more sample per split, model is less likely to create
    splits based on noise in the training data. 
        - 2 is needed for split, and a maximum of 20 since, 20/150 ~ 13% (150
        samples) where common practice is splitting between 1% - 15%.

    "min_samples_leaf" -> Samples required at each leaf node w/ higher values
     creating more robust predictions (averaging over more samples), which
     helps prevent overfitting, and controls what happens AFTER the split.
        - min of 1: Highest granularity (more detailed decision boundaries)
        - max of 10: 50 samples per class, where 10 ~ 20% of each class.

    "criterion" -> criteria=False (False: "gini", ,True: "entropy")
        Gini: Probability of incorrect classifications 
            - Computationally more efficient
            - Range between 0 and 1, isolating more frequent class in branch
        Entropy: Information needed to encode class distribution
            - Creates more balanced trees
            - Range between 0 and log2(n_classes)
    ----------------------------------------------------------------
    INPUT: Fill these in?
        
    OUTPUT:
    """
    return {"max_depth": max_depth, "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf, "criterion": criterion}

# TASK 1: =======================================================
# Load dataset
def load_split_data():
    """
    Loads both classification (iris) and regression (california) datasets and
    splits them.
    ------------------------------------------------------------
    X_cls: Features (sepal length, sepal width, petal length, petal width)
            y_cls: Target Labels (species of the iris flower: 0, 1, or 2)
    X_reg: Features (median income, house age)
    y_reg: Target variable (median house value)
    Data split according to the TEST_SIZE, which is 20% of the data for
    testing.
    ------------------------------------------------------------
    INPUT:
        None

    OUTPUT:
        (X_cls_train, X_cls_test, y_cls_train, y_cls_test,
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
        iris, california): (tuple) All trainiing and test sets for both
        datasets, along with original dataset objects.
    """
    # Classification data
    iris = load_iris()
    X_cls, y_cls = iris.data, iris.target

    # Hyperparameter grid setup +++++++++++++++++++++++++++++++++++++++
    # Split data
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls,
                                                                        y_cls,
                                                                        test_size=TEST_SIZE,
                                                                        # Reproducibility
                                                                       random_state=RANDOM_STATE)

    # Regression data
    california = fetch_california_housing()
    X_reg, y_reg = california.data, california.target
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg,
                                                                       y_reg,
                                                                        test_size=TEST_SIZE)

    return (X_cls_train, X_cls_test, y_cls_train, y_cls_test,
            X_reg_train, X_reg_test, y_reg_train, y_reg_test,
            iris, california)

# Random Search
def perform_random_search(X_train, y_train, params):
    """
    Implements RandomizedSearchCV to find optimal parameters for a Decision
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
    Performs classification analysis including the Confusion matrix.
    -------------------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nClassification Metrics: ")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-SCore: {f1:.4f}")

    print("\nConfusion Matrix Analysis: ")
    print("-"*50)
    print("Metrics for each class ")
    
    n_classes = len(class_names)
    for i in range(n_classes):
        # True positives: Correct predictions current class
        tp = cm[i, i]
        # False positives: Other classes predicted as current class
        fp = np.sum(cm[:, i]) - tp
        # False Negatives: Current class predicted as other classes
        fn = np.sum(cm[i, :]) - tp
        # True Negatives: Correct predictions for all other classes
        tn = np.sum(cm) - (tp + fp + fn)

        print(f"\nClass {class_names[i]}:")
        print(f"True Positives (TP): {tp} - Correctly predicted {class_names[i]}")
        print(f"""True Negatives (TN): {tn} - Correctly predicted other classes""")
        print(f"""False Positives (FP): {fp} - Incorrectly predicted
              {class_names[i]}""")
        print(f"""Fales Negatives (FN): {fn} - {class_names[i]} Incorrectly
              predicted other classes""")

        # Class-specific metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Sensitivity (TPR): {sensitivity:.4f}")
        print(f"Specificity (TNR): {specificity:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
#    plt.show() ?

    return accuracy, precision, recall, f1

def train_evaluate_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains and evaluates Decision Tree Regression model.
    --------------------------------------------------------
    INPUT:
        X_train: (np.ndarray) Training Data
        X_test: (np.ndarray) Test data
        y_train: (np.ndarray) Training labels
        y_test: (np.ndarray) True test labels
        feature_names: ()

    OUTPUT:
        mse: ()
        rmse: ()
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
#    plt.show() ?

    return (mse, rmse), dt_reg

def compare_models(cls_metrics, reg_metrics, cls_model, reg_model, iris,
                   california, X_reg_test, y_reg_test):
    """
    Compares performance of classification and regression models.
    --------------------------------------------------------
    INPUT:
        cls_metrics: ()
        reg_metrics: ()

    OUTPUT:
    """
    print("\nModel Comparison")
    print("="*50)

    # Classification metrics
    print("Classification Model Performance")
    print(f"- Accuracy: {cls_metrics[0]:.4f}")
    print(f"- Precision: {cls_metrics[1]:.4f}")
    print(f"- Recall: {cls_metrics[2]:.4f}")
    print(f"- F1 Score: {cls_metrics[3]:.4f}")

    # Feature importance for classification with error handling
    try:
        cls_importance = pd.DataFrame({
            "feature": list(iris.feature_names),  # Convert to list explicitly
            "importance": list(cls_model.feature_importances_)
        })
        cls_importance = cls_importance.sort_values("importance", ascending=False)
        print("\nTop Classification Features:")
        print(cls_importance)
    except ValueError as e:
        print(f"\nError creating classification importance DataFrame: {e}")
        print("Feature names:", iris.feature_names)
        print("Feature importances:", cls_model.feature_importances_)

    # Regression metrics
    print("\nRegression Model Performance (California Housing)")
    print("="*50)
    print(f"- MSE: {reg_metrics[0]:.4f}")
    print(f"- RMSE: {reg_metrics[1]:.4f}")
    print(f"R^2 Score: {reg_model.score(X_reg_test, y_reg_test):.4f}")

    # Feature importance for regression with error handling
    try:
        reg_importance = pd.DataFrame({
            "feature": list(california.feature_names),  # Convert to list explicitly
            "importance": list(reg_model.feature_importances_)
        })
        reg_importance = reg_importance.sort_values("importance", ascending=False)
        print("\nTop Regression Features:")
        print(reg_importance)
    except ValueError as e:
        print(f"\nError creating regression importance DataFrame: {e}")
        print("Feature names:", california.feature_names)
        print("Feature importances:", reg_model.feature_importances_)

    print("\n1. Classification Model:")
    print("- Model's performance evaluated through accuracy and f1-score")
    print("- High precision/recall indicates good balance between false positives and negatives")

    print("\n2. Regression Model:")
    print("- RMSE provides error metric in same units as target variable")
    print("- Lower RMSE indicates better model performance")

def analyze_best_model(X_train, y_train, iris, params):
    """
    Analyzes best model, outputting best parameters, cross-validatoin score,
    and model.
    --------------------------------------------------------------
    INPUT:
        X_train: (np.ndarray) Training data
        y_train: (np.ndarray) Training labels

    OUTPUT:
        best_model: ()
    """
    random_search = perform_random_search(X_train, y_train, params)
    print("\nModel Analysis Results")
    print("-"*50)
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

def error_analysis2(best_model, X_test, y_test, iris):
    """
    Analyzes misclassified instances from the decision tree model.
    ------------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    # Predictions
    y_prediction = best_model.predict(X_test)

    # Find misclassified instances
    misclassified_mask = y_test != y_prediction
    misclassified_indices = np.where(misclassified_mask)[0]

    print("\nError Analysis Results: ")
    print("-"*50)
    print(f"Number of misclassified instances: {len(misclassified_indices)}")
    print(f"Test set accuracy: {accuracy_score(y_test, y_prediction):.4f}")

    if len(misclassified_indices) > 0:
        print("\nMisclassified Instances: ")

        for idx in misclassified_indices:
            print(f"\nIndex: {idx}")
            print(f"True Class: {iris.target_names[y_test[idx]]}")
            print(f"Predicted class: {iris.target_names[y_prediction[idx]]}")
            print("Feature: ")

            for feature_name, value in zip(iris.feature_names, X_test[idx]):
                print(f"\t{feature_name}: {value:.2f}")

                # Decision path
                path = best_model.decision_path([X_test[idx]])

                print("\nDecision path for features: ")

                for node_id in path.indices:
                    
                    if (node_id < len(best_model.tree_.feature) and
                    # -2 indicates leaf
                    best_model.tree_.feature[node_id] != -2):
                        feature = \
                        iris.feature_names[best_model.tree_.feature[node_id]] 
                        threshold = best_model.tree_.threshold[node_id]
                        print(f"""Split on {feature} at threshold
                              {threshold:.2f}""")

        return misclassified_indices

def error_analysis1(best_model, X_test, y_test, iris):
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
#    plt.show() ?

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
#        plt.show() ?

def main():
    # Load split data and get hyperparameters
    (X_cls_train, X_cls_test, y_cls_train, y_cls_test, X_reg_train, X_reg_test,
    y_reg_train, y_reg_test, iris, california) = load_split_data()
    params = define_params() # Using default values

    # Task 1: Best classification model ==============================
    # Classification analysis
    best_cls_model = analyze_best_model(X_cls_train, y_cls_train, iris, params)

    # Task 2: Error Analysis ============================== 
    misclassified_indices = error_analysis2(best_cls_model, X_cls_test, y_cls_test, iris)

    # Task 3: Confusion Matrix ==============================
    cls_prediction = best_cls_model.predict(X_cls_test)
    cls_metrics = analyze_classification_results(y_cls_test, cls_prediction,
                                                 iris.target_names)

    # Task 4: Train/evaluate regression model==============================
    reg_metrics, reg_model = train_evaluate_regression(X_reg_train, X_reg_test,
                                                      y_reg_train, y_reg_test,
                                                      california.feature_names)
    
    # Task 5: Compare Models ==============================
    cls_metrics = (
        accuracy_score(y_cls_test, cls_prediction),
        precision_score(y_cls_test, cls_prediction, average="weighted"),
        recall_score(y_cls_test, cls_prediction, average="weighted"),
        f1_score(y_cls_test, cls_prediction, average="weighted")
    )

    compare_models(cls_metrics, reg_metrics, best_cls_model, reg_model, iris,
                  california, X_reg_test, y_reg_test) 

    # Experimentation ... DELETE ??
    breakpoint()


if __name__ == "__main__":
    main()
