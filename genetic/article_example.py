#!/usr/bin/env python

# Data
knapsack_data = {
    "capacity": 20,
    "items": [
        {"weight": 1, "value": 3},
        {"weight": 3, "value": 5},
        {"weight": 4, "value": 12},
        {"weight": 5, "value": 15},
        {"weight": 6, "value": 18},
        {"weight": 8, "value": 20}
    ],
    "chromosomes": [
        {"bits": [0, 0, 1, 0, 1, 1], "weight": 18, "value1": 50},
        {"bits": [0, 1, 1, 0, 0, 1], "weight": 15, "value2": 37},
        {"bits": [1, 1, 1, 1, 1, 0], "weight": 19, "value3": 53},
        {"bits": [0, 0, 0, 1, 0, 1], "weight": 13, "value4": 35}
    ]
}

def get_initial_population(data):
    """
    generates initial population.
    ------------------------------------------------
    input:
        data: (dict)

    output:
        pop_size: (array) population of individuals.
        generation: (int) current generation.
        capacity: (int) knapsack capacity.
        S: (lst of tuples) set of n items with weights and values.
        stop: (int) final generation (stopping condition).
    """
    capacity = data["capacity"]
    population = [data["chromosomes"][i]["bits"] for i in
                  range(len(data["chromosomes"]))]
    n = len(population[0]) # Number of items (genes)
    pop_size = len(population)
    generation = 0 # Initial generation

    # Set of items: (weights, values)
    S = [(item["weight"], item["value"]) for item in data["items"]]
    # Stopping criteria
    stop = 10 # No reason

    return population, capacity, S, generation, stop
    

breakpoint()
