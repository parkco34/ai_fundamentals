#!/usr/bin/env python


def get_initial_population(config):
    """
    PROVIDED VIA INSTRUCTIONS.
    Populates variables from config and initializes P at gen 0.
    ------------------------------------------------------------
    INPUT:
        config: (str) Path to config file.

    OUTPUT:
        generation: (int)  Current generation.
        population: (np.narray) Population
        W: (int) Knapsack capacity.
        S: (list of tuples) Each ordered pair (w_i: weight of ith item, v_i: value if ith item)
        stop: (int) Final generation (stopping condition)
    """
    # Set random seed
    np.random.seed(1473)
    with open(config, "r") as file:
        lines = file.readlines()

    # Converts the genes to integer values (0 and 1)
    pop_size, n, stop, W = map(int, [lines[i].strip() for i in range(4)])
    S = [tuple(map(int, line.strip().split())) for line in lines[4:]]

    generation = 0
    # Random initialization of population
    population = np.random.randint(2, size=(pop_size, n))

    return population, W, S, generation, stop

# Info
population, capacity, S, generation, stop = get_initial_population("data/config_1.txt")




breakpoint()
    

