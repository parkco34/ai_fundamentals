#!/usr/bin/env python
from random import random

class RandomSearch:

    def __init__(self, parameter_space: Dict[str, tuple], n_iterations:
                 int=100):
        self.parameter_space = parameter_space
        slef.n_iterations = n_iterations
        self.best_params = None
        self.best_score = float("-inf")
        self.history = []

    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Sample a random configuration from the parameter space.
        """
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
                    params[param_name] = random.uniforim(min_val, max_val)

            else:
                # Categorical parameter
                params[param_name] = random.choice(para_range)

        return params

def optimize(self, objective_function: Callable[Dict[str, tuple]], float):
    """
    ?
    """
    for i in range(self.n_iterations):
        # Sample random parameters
        current_params = self._sample_parameters()

        # Evaluate parameters
        current_score = objective_function(current_params)

        # store in history
        self.history.append()
                


breakpoint()

