#!/usr/bin/env python
import logging
from random import uniform, sample, randint

# Setting up logging
logging.basicConfig(filename="example_population.log", level=logging.INFO,
                    format="%(message)s")

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

# Population size determined via "small" for simple problem
POP_SIZE = 50
ITEMS = [(item["weight"], item["value"]) for item in knapsack_data["items"]]

def get_data(config_file):
    """
    Generates the initial population.
    ------------------------------------
    INPUT:
        config_file: (str) Path to configuration file

    OUTPUT:
        population: (2d array)
        capacity: (int) Max weight of knapsack
        items: (list of tuples) weight-value pairs
        stop: (int) Stopping condition
    """
    pass

def get_population(data):
    """
    BINARY ENCODING: Randomly generate UNIQUE binary strings for the number of items,
    where 1 represents item being included in knapsack and 0 for exclusion of
    item.
    ------------------------------------------
    INPUT:
        data: (list of dictionaries) Dict items in "items" key

    OUTPUT:
        population: (list)
    """
    # Ensure unique individuals
    unique_population = set()
    
    for _ in range(POP_SIZE * 2):
        chromosome = tuple(randint(0, 1) for _ in range(len(data)))
        unique_population.add(chromosome)

        if len(population) >= POP_SIZE:
            break

    # Convert to list of lists
    population = [list(chromosome) for chromosome in unique_population]

    # Log population for verification of my results
    logging.info("Generated Population:")
    for chromosome in population:
        logging.info(f"{chromosome}")

    return population

def fitness(chromosome):
    """
    Calculates fitness of chromosome, ensuring it doesn't exceed maximum
    capacity of knapsack, returning the total value of chromosome.
    ------------------------------------------------------
    INPUT:
        chromosome: (list) Individual in population

    OUTPUT:
        total_value: (float)
    """
    total_weight = sum(gene * item[0] for gene, item in zip(chromosome, ITEMS))
    total_value = sum(gene * item[1] for gene, item in zip(chromosome, ITEMS))

    if total_weight > CAPACITY:
        return 1e-5

    return total_value

def roulette_selection(population):
    """
    GENERATES TWO MOST FIT PARENTS.
    1. Generate and sort fitness values
    2. Relative fitnesses 
    3. Cumulative fitnesses
    4. Spin wheel: 
        If uniform random variable (0,1) is contained within cumulative
        fitness, selecting first individual whose cumulative fitness is greater
        than or equal to the uniform random number.
    ----------------------------------------
    INPUT:
        population: (list) All current chromosomes

    OUTPUT:
        parents: (tuple)
    """
    fitness_scores = [fitness(chrome) for chrome in population]
    total_fitness = sum(fitness_scores)
    relative_scores = [fitness_scores[i] / total_fitness for i in
                       range(len(fitness_scores))]
    cumulative_scores = [sum(relative_scores[:i+1]) for i in
                         range(len(relative_scores))]

    # Uniform random number: "Spining" the wheel
    rand_num = uniform(0,1)

    parents = () # Intialize parents tuple
    for i, prob in enumerate(cumulative_scores):
        # "Spinning" the wheel
        if point <= prob:
            # Ensure unique parents
            if population[i] not in parents:
                parents.append(population[i])

            break

    return parents

def tournament_selection():
    """
    GENERATES TWO MOST FIT PARENTS.
    1. Choose tournament size
    2. Randomly select a number of individuals from population, selecting the
    most fit among them to be parents for next generation.
     - Selection Pressure adjusted via TOURNAMENT SIZE.
    3. Determine fittest individual from group, based on their fitness.
    4. Repeat until enough parents.
    ----------------------------------------
    INPUT:
        population: (list) All current chromosomes
        tournament_size: (int) Number individuals in tournament at a time.

    OUTPUT:
        parents: (tuple)
    """
    pass

def single_point_crossover():
    pass

def k_point_crossover():
    pass

def flip_bit_mutation():
    pass

def main():
    pass


breakpoint()
