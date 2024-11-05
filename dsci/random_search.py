#!/usr/bin/env python
from random import random, randint, uniform, choice
import matplotlib.pyplot as plt

def initialize():
    """
    Initializes required variables for Random Search.
    ------------------------------------------------------------
    INPUT:

    OTPUT:
        best_params: (dict) Best parameters found
        best_score: (float) Best score achieved
        history: (list) History of all iterations
    """
    return None, float("-inf"), []

def sample_params(param_space):
    """
    Sample random configuration from parameter space.
    --------------------------------------------------------
    INPUT:
        param_space: (dict) Dictionary defining parameter ranges.

    OUTPUT:
        params: (dict) Sampled parameters
    """
    params = {}

    for param_name, param_range in param_space.items():

        if isinstance(param_range, tuple):
            # Numerical parameter
            min_val, max_val = param_range

            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                params[param_name] = randint(min_val, max_val)

            else:
                # Float parameter
                params[param_name] = uniform(min_val, max_val)

        else:
            # Categorical parameter
            params[param_name] = choice(param_range)

    return params

def optimize(objective_function, param_space, best_params, best_score, history, n_iterations):
    """
    Runs Random Search optimization.
    ------------------------------------------------------------
    INPUT:
        objective_function: (function) Function to optimize
        param_space: ()
        best_score:  (float)
        history: (list)
        n_iterations: (int)

    OUTPUT:
        best_params: (dict) Best parameters found
        best_score: (float) Best score achieved
        history: (list) Updated history including all iterations
    """
    data = get_data("random_search_data.txt")

    for i in range(n_iterations):
        # Sample random parameters
        current_params = sample_params(param_space)

        # Evalutate parameters
        current_score = objective_function(current_params, data)

        # Store in history
        history.append({
            "iteration": i,
            "parameters": current_params,
            "score": current_score
        })
        
        # Update best parameters if current score is better
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()

        print(f"Iteration {i+1}/{n_iterations}, Score: {current_score:.4f}, Best Score: {best_score:.4f}")


    return best_params, best_score, history

def objective_function(params, data):
    """
    Negative Mean Squared Error for given parameters.
    --------------------------------------------------------------
    INPUT:
        params: (dict) Parameters 'a' and 'b' for linear model

    OUTPUT:
        score: (float) Negative mean squared error
    """
    # Parameter ranges
    a = params["a"]
    b = params["b"]
    
    # x indices
    x_values = data["x"]
    # target values
    y_values = data["y"]
    
    y_prediction = [a * x + b for x in x_values]
    # Mean Squared Error (MSE)
    mse = sum((y_p - y_t)**2 for y_p, y_t in zip(y_prediction, y_values)) / \
    len(y_values)

    return -mse

def get_data(filename):
    """
    Populates dictionary by first extracting the column headers, then
    populates key-value pairs 
    ------------------------------------------------------------
    INPUT:
        filename: (Str)

    OUTPUT:
        data: (dict) Dataset
    """
    # Read file
    with open("random_search_data.txt", "r") as file:
        text = file.read()

        # strip removes any trailing white space
        lines = text.strip().split("\n")
        # First line is column headers, splitting divide
        columns = lines[0].split()
        
        # Initialize dictionary
        data = {column: [] for column in columns}

        # populate dictionary, from second line down
        for line in lines[1:]:
            # Splits the line (str) into a list of string-values
            values = line.split()
            # Zip up columns into a list of ORDERED PAIRS
            for column, value in zip(columns, values):
                data[column].append(float(value))

    return data

# Ranges
param_space = {
    "a": (0.5, 1.5),
    "b": (1.5, 2.5)
}

n_iterations = 1000

best_params, best_score, history = initialize()

# Unr optimization
best_params, best_score, history = optimize(objective_function, param_space,
                                            best_params, best_score, history,
                                            n_iterations)

# Output the best parameters
print("\nOptimization Completed.")
print(f"Best Parameters: {best_params}")
print(f"Best Score (Negative MSE): {best_score:.4f}")
# Assuming history contains the score of each iteration
cumulative_best = []
best_so_far = float("-inf")
for entry in history:
    best_so_far = max(best_so_far, entry['score'])
    cumulative_best.append(best_so_far)

plt.plot(cumulative_best)
plt.title('Cumulative Best Score over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Best Negative MSE')
plt.show()

