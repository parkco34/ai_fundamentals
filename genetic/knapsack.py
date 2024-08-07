#!/usr/bin/env python


def get_initial_population(config):
    """
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


    

