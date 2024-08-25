#!/usr/bin/env python
"""
0/1 Knapsack Problem -Completed: 8/25/24
-----------------------------------------------------------------------------
-------------
Example usage:
-------------
data = {
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

POP_SIZE = 50
ITEMS = [(item["weight"], item["value"]) for item in knapsack_data["items"]]
CAPACITY = knapsack_data["capacity"]
population = get_population(data["items"], POP_SIZE)
-----------------------------------------------------------------------------
"""
import logging
import numpy as np
import random
from random import uniform, sample, randint

# Setting up logging
logging.basicConfig(filename="example_population.log", level=logging.INFO,
                    format="%(message)s")

def generate_seed():
    return randint(0, 2 ** 32 - 1)

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
    with open(config_file, "r") as file:
        lines = file.readlines()
        pop_size = int(lines[0].strip())
        n = int(lines[1].strip())
        stop = int(lines[2].strip())
        W = int(lines[3].strip())
        S = [tuple(map(int, line.strip().split())) for line in lines[4:]]

    return pop_size, n, stop, W, S

def get_population(data, pop_size):
    """
    BINARY ENCODING: Randomly generate UNIQUE binary strings for the number of items,
    where 1 represents item being included in knapsack and 0 for exclusion of
    item.
    ------------------------------------------
    INPUT:
        data: (list of dictionaries) Dict items in "items" key
        pop_size: (int) Population size

    OUTPUT:
        population: (list)
    """
    # INput validation
    if not (isinstance(data, list) and isinstance(data[0], dict)):
        raise ValueError("""{20 * "*"}\n\nInvalid input!\n\nInput needs to be a list of
                         dictionaries\n\n{20 * "*"}""")

    # Getting unique population 
    population = []
    while len(population) < pop_size:
        chromosome = [randint(0,1) for _ in range(len(data))]

        # Ensure uniqueness
        if chromosome not in population:
            population.append(chromosome)

    # Log population
    logging.info("Generated Population")
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
        total_value: (int) 
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
        parents: (list)
    """
    # Caculate fitness scores for each chromosome
    fitness_scores = [fitness(chrome) for chrome in population]
    # Relative/cumulative fitness for constructing the wheel
    total_fitness = sum(fitness_scores)

    # Avoid division by zero
    if total_fitness == 0:
        raise ValueError("Total fitness is zero; can't perform roulette selection.")
    relative_scores = [score / total_fitness for score in fitness_scores if
                       total_fitness != 0]
    cumulative_scores = [sum(relative_scores[:i+1]) for i in
                         range(len(relative_scores))]

    parents = []

    while len(parents) < 2:
        points = sorted([uniform(0,1), uniform(0,1)])
        # Uniform random number: "Spining" the wheel; selecting first and second
        # parent
        for i, prob in enumerate(cumulative_scores):
            if len(parents) < 2 and prob >= points[len(parents)]:
                parents.append(population[i])

    return parents[0], parents[1]

def tournament_selection(population, p=0.75, tournament_size=3):
    """
    GENERATES MOST FIT PARENTS FOR REPRODUCTION VIA COMPETITION
    -----------------
    LIKELIHOOD VALUES*
    -----------------
    HIGH (p <= .9):
        - Strongly favors most fit in tournament
        - Speeds up convergence by consistently selecting best individuals
    MEDIUM (0.70 <= p <= 0.8):
        - Maintains diversity by allowing some less-fit individuals to be
        chosen in next generation
        - Good starting point if you're not sure about best value for the
        p*
    LOW (p <= 0.6):
        - More randomness
        - Prevents premature convergence
    ----------------------------------------
    1. Choose tournament size
    2. Randomly select a number of individuals from population, selecting the
    most fit among them to be parents for next generation.
     - Selection Pressure adjusted via TOURNAMENT SIZE.
    3. Determine fittest individual from group, based on their fitness.
    4. Repeat until enough parents.
    ----------------------------------------
    INPUT:
        population: (list) All current chromosomes
        p: (float; default=0.75) Probability of selecting the most highly ranked
        indiviudal.
        tournament_size: (int; default=3) Number individuals in tournament at a time.

    OUTPUT:
        parent1, parent2: (list of lists)
    """
    parents = []
    
    # Tournament size not greater than population size
    tournament_size = min(tournament_size, len(population))

    while len(parents) < 2:
        contestants = sample(population, tournament_size)
        contestants.sort(key=fitness, reverse=True)

        # Select best individual with probability p
        if random() < p:
            winner = contestants[0]

        else:
            # Not best
            winner = contestants[randint(1, tournament_size - 1)]
        
        # Uniqueness of parents
        if winner not in parents:
            parents.append(winner)

    if len(parents) < 2:
        parents.append(parents[0])

    return parents[0], parents[1]

def valid_solution(chromosome, items, capacity): # ?
    """
    Decides whether the resulting offspring are valid solutions.
    -----------------------------------
    INPUT:
        chromosome: (list) Individual
        items: (list of tuples) weight-value pairs
        capacity: (int) Maximum weight of knapsack

    OUTPUT:
        (bool) Valid solution or not.
    """
    total_weight = sum(chromosome[i] for i in range(len(chromosome)))
    return total_weight <= capacity

# Crossover
def single_point_crossover(parent1, parent2, crossover_rate=0.7,
                           mutation_rate=0.01):
    """
    Single-point crossover between two abusive parents, splitting at a random
    point in the chromosome, where the the genes after that point will be
    swapped with the other parent.
    ----------------------------------------------------
    INPUT:
        parent1: (list) 
        parent2: (list)
        crossover_rate: (float; default=0.7) Probability of 
        mutation_rate: (float; default=0.01) Probability of mutating a bit

    OUTPUT:
        child1, child2: (list of lists) Offspring
    """
    children = []

    # Crossover or not
    if random() <= crossover_rate:
        # Crossover point, where we don't include the first and last index,
        # ensuring crossover is fruitful, avoiding asexaul reproduction
        point = randint(1, len(parent1)-1) 
        
        # Crossover
        child1, child2 = parent1[:point] + parent2[point:], parent2[:point] + \
        parent1[point:]

    else:
        child1, child2 = parent1[:], parent2[:]

    # Mutants
    child1 = bit_flip_mutation(child1)
    child2 = bit_flip_mutation(child2)

    return child1, child2

def k_point_crossover():
    pass

# Mutation
def bit_flip_mutation(child, mutation_rate=0.01):
    """
    Mutation of child via BIT-FLIP, determined by the mutationrate:
    ~ 1 / len(child).
    ---------------------------------------------
    INPUT:
        child: (list)
        mutation_rate: (float; default=0.01) Probability of mutating a bit

    OUTPUT:
        mutated_child: (list)
    """
    if isinstance(child, list):
        # Copy child
        mutated_child = child[:]
        
        for i in range(len(mutated_child)):
            if random() <= mutation_rate:
                # FLips bit
                mutated_child[i] = 1 - mutated_child[i]

    return mutated_child

# ?
def create_new_population(population, elite_size=2):
    """
    Generates a new population via elitism ...
    ---------------------------------------------
    INPUT:
        population: (list)
        elite_size: (int; default=2) Number of the best fit individuals to
        carry on to the next generation, w/out any crossover or mutation

    OUTPUT:
        new_population: (list of lists) For next generation
    """
    # Sort based on fitness
    sorted_population = sorted(population, key=fitness, reverse=True)
    
    # Elists
    new_population = sorted_population[:elite_size]
    
    # Iteratively generate new population of individuals
    while len(new_population) < len(population):
        # ? --> Replace specific function with parameter
        parent1, parent2 = tournament_selection(sorted_population)
        child1, child2 = single_point_crossover(parent1, parent2)

        if len(new_population) < len(population):
            new_population.append(child1)
            new_population.append(child2)

    # ? --> Input validation
    
    return new_population 

def main(config_file, seed=None):
    if seed is None:
        seed = generate_seed()

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
        
    print(f"Using random seed: {seed}")

    # COnstants
    global POP_SIZE, ITEMS, CAPACITY
    POP_SIZE, N, STOP, CAPACITY, ITEMS = get_data(config_file)

    population = get_population(ITEMS, POP_SIZE)

    generation = 0
    max_generations = 100

    breakpoint()
    while generation < max_generations:
        population = create_new_population(population)
        best_solution = max(population, key=fitness)
        logging.info(f"""Generation {generation}: Best solution {best_solution}
                    with fitness {fitness(best_solution)}""")
        generation += 1

    best_solution = max(population, key=fitness)
    print(f"Best solution is {fitness(best_solution)}")
    print(f"Winner: {best_solution}")


if __name__ == "__main__":
    main("data/config_1.txt")
