#!/usr/bin/env python
"""
Part 2: Support Vector Machine (SVM) - Linear and w/ RBF Kernel.
5pts. each task
Please complete the following tasks to explore Linear Support Vector Machines (LSVM) and Support
Vector Machines (SVM) with an RBF kernel. You can use Python and libraries such as scikit-learn to
implement and demonstrate your work.
2
Task 1: Linear Support Vector Machine (LSVM)
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a Linear Support Vector Machine (SVM) classiWier using scikit-learn.
4. Train the LSVM model on the training data.
5. Evaluate the LSVM model’s performance on the test data and report accuracy.
Task 2: Support Vector Machine (SVM) with RBF Kernel
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Implement a Support Vector Machine (SVM) classiWier with an RBF kernel using scikit-learn.
4. Train the SVM model with the RBF kernel on the training data.
5. Evaluate the SVM model’s performance on the test data and report accuracy.
Task 3: Hyperparameter Tuning for SVM with RBF Kernel
1. Perform hyperparameter tuning for the SVM with an RBF kernel. Search for optimal values of
hyperparameters such as C and γ using Random Search.
2. Report the best hyperparameters for the SVM with the RBF kernel.
3. Train a new SVM model with the best hyperparameters and evaluate its performance on the test
data.
Task 4: Metrics Comparison
1. Calculate and compare relevant evaluation metrics (e.g., accuracy, precision, recall, F1-score) for
the LSVM from Task 1 and the SVM with an RBF kernel from Task 2.
2. Not graded: Discuss the differences in performance and characteristics between these models.
"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def linear_svm(X_train, X_test, y_train, y_test):
    """
    Implements and evaluates Linear SVM classifier.
    --------------------------------------------------------
    INPUT:
        X_train: (np.ndarray) Training data
        X_test: (np.ndarray) Training Labels
        y_train: (np.ndarray) 

    OUTPUT:
    """
    # Initialize Linear SVM
    linear_svm = SVC(kernel="linear", random_state=RANDOM_STATE)

    # Train model
    linear_svm.fit(X_train, y_train)

    # Prediction
    y_prediction = linear_svm.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_prediction)
    print("\Linear SVM results: ")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")

    return linear_svm, accuracy, y_prediction

def rbf_svm(X_train, X_test, y_train, y_test):
    """
    SVM classifier with RBF kernel (Radial Basis Function Kernel)
    -----------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    # Initialize RBF SVM
    rbf_svm = SVC(kernel="rbf", random_state=RANDOM_STATE)

    # Training model
    rbf_svm.fit(X_train, y_train)

    # Predictions
    y_prediction = rbf_svm.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_prediction)
    print("\nRBF SVM Results:")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")

    return rbf_svm, accuracy, y_prediction

def fine_tuning(X_train, X_test, y_train, y_test):
    """
    Performs hyperparameters tuning for SVM with RBF kernel using
    RandomizedSearchCV.
    ------------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    # Paramter distribution for random search
    param_dist = {
        "C": loguniform(1e-3, 1e3),
        "gamma": loguniform(1e-4, 1e1)
    }

    # Initialize base model
    base_model = SVC(kernel="rbf", random_state=RANDOM_STATE)

    # Initialize random search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distribution=param_dist,
        n_iter=100,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    # Perform random search
    random_search.fit(X_train, y_train)
    
    # get best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Predictions with best model
    y_prediction = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)

    print("\nTuned RBF SVM Results: ")
    print("="*50)
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-validation Score: {random_search.best_score_:.4f}")
    print(f"Tet Accuracy: {accuracy:.4f}")

    return best_model, best_params, accuracy, y_prediction

def compare_models(models_data, y_test, iris):
    """
    Compares performance metrics of different SVM models.
    -------------------------------------------------------
    INPUT:

    OUTPUT:
    """
    metrics = {}

    for model_name, (y_pred, _) in models_data.items():
        metrics[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted")
        }

    # Comparison DataFrame
    df = pd.DataFrame(metrics).round(4)

    print("\nModel Comparison: ")
    print("="*50)
    print(df)

    # Visualize confusion matrix
    fig, axes = plt.subplots(1, len(models_data), figsize=(15, 5))
    for i, (model_name, (y_pred, _)) in enumerate(models_data.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, format="d", ax=axes[i],
                   xticklabels=iris.target_names,
                   yticklabels=iris.target_names)
        axes[i].set_title(f"{model_name}\nConfusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout()
    plt.show()

    return df

def main():
    # Load and split data
    (X_csl_train, X_cls_test, y_cls_train, y_cls_test, _, _, _, _, iris, _) = \
    load_split_data()

    # Task1: Linear SVM
    linear_model, linear_acc, linear_pred = linear_svm(X_cls_train, X_cls_test,
                                                      y_cls_train, y_cls_test)

    # Task 2: RBF SVM
    rbf_model, best_params, tuned_acc, tuned_pred = fine_tuning(X_cls_train,
                                                                X_cls_test,
                                                                y_cls_train,
                                                                y_cls_test)

    # Task 3: Fine tuning RBF SVM
    tuned_model, best_params, tuned_acc, tuned_pred = fine_tuning(X_cls_train,
                                                                 X_cls_test,
                                                                  y_cls_train,
                                                                 y_cls_test)

    # Task 4: Compare Models
    modes_data = {
        "Linear SVM": (linear_model, linear_acc),
        "RBF SVM": (rbf_pred, rbf_acc),
        "Tuned RBF SVM": (tuned_pred, tuned_acc)
    }

    df = compare_models(models_data, y_cls_test, iris)



if __name__ == "__main__":
    main()



