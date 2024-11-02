#!/usr/bin/env python
import random
from typing import Dict, List, Callable, Any

class RandomSearch:
    def __init__(self, parameter_space: Dict[str, tuple], n_iterations: int = 100):
        """
        Initialize Random Search optimizer.

        Args:
            parameter_space: Dictionary where keys are parameter names and values are tuples
                           defining the range (min, max) for numerical parameters or
                           list of choices for categorical parameters
            n_iterations: Number of random samples to try
        """
        self.parameter_space = parameter_space
        self.n_iterations = n_iterations
        self.best_params = None
        self.best_score = float('-inf')
        self.history = []

    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample a random configuration from the parameter space."""
        params = {}
        for param_name, param_range in self.parameter_space.items():
            if isinstance(param_range, tuple):
                # Numerical parameter
                min_val, max_val = param_range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    # Float parameter
                    params[param_name] = random.uniform(min_val, max_val)
            else:
                # Categorical parameter
                params[param_name] = random.choice(param_range)
        return params

    def optimize(self, objective_function: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run random search optimization.

        Args:
            objective_function: Function that takes a parameter configuration and returns a score to maximize

        Returns:
            Best parameter configuration found
        """
        for i in range(self.n_iterations):
            # Sample random parameters
            current_params = self._sample_parameters()

            # Evaluate parameters
            current_score = objective_function(current_params)

            # Store in history
            self.history.append({
                'iteration': i,
                'parameters': current_params,
                'score': current_score
            })

            # Update best parameters if current score is better
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_params = current_params.copy()

            print(f"Iteration {i+1}/{self.n_iterations}, Score: {current_score:.4f}, Best Score: {self.best_score:.4f}")

        return self.best_params

# Example usage
def example():
    # Define parameter space
    param_space = {
        'learning_rate': (0.0001, 0.1),
        'num_layers': (1, 5),
        'activation': ['relu', 'tanh', 'sigmoid'],
        'dropout': (0.1, 0.5)
    }

    # Define objective function (dummy example)
    def objective_function(params):
        # This would normally be your model training and validation
        # Here we just create a dummy score
        score = (0.1 / params['learning_rate']) + params['num_layers'] * 2
        if params['activation'] == 'relu':
            score *= 1.2
        score *= (1 - params['dropout'])
        return score

    # Initialize and run random search
    rs = RandomSearch(param_space, n_iterations=20)
    best_params = rs.optimize(objective_function)

    print("\nOptimization completed!")
    print(f"Best parameters found: {best_params}")
    print(f"Best score achieved: {rs.best_score:.4f}")

    return rs

if __name__ == "__main__":
    example()

