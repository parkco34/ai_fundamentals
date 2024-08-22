#!/usr/bin/env python
from random import sample, randint

# Define knapsack data first
knapsack_data = {
    "capacity": 20,
    "items": [
        {"weight": 1, "value": 3},
        {"weight": 3, "value": 5},
        {"weight": 4, "value": 12},
        {"weight": 5, "value": 15},
        {"weight": 6, "value": 18},
        {"weight": 8, "value": 20}
    ]
}

# Number of genes in chromosome = # of ITEMS
GENE_SIZE = len(knapsack_data["items"])
# of chromosomes arbitrarily chosen depending on stuff ?
POPULATION_SIZE = 10

# Initialize population
def get_initial_population(data):
    """
    Genrate initial population via
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
    capacity = data["capacity"] # Max weight
    # Randomly generate genes for each chromosome 0 or 1
    population = [[randint(0,1) for _ in range(GENE_SIZE)] for _ in
                  range(POPULATION_SIZE)]
    generation = 0 # Initial generation

    # Set of items: (weights, values)
    S = [(item["weight"], item["value"]) for item in data["items"]]
    # Stopping criteria
    stop = 10 # No reason ? Dynamic programming to handle this automatically...

    return population, capacity, S, generation, stop




breakpoint()
