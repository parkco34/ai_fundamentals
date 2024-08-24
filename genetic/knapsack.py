#!/usr/bin/env python
import logging
from random import uniform, sample, randint, random

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
CAPACITY = knapsack_data["capacity"]

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
    cumulative_prob = 0
    remaining_prob = 1
    count = 0
   
    # If tournament size None, include 20% of the population
    if not tournament_size:
        tournament_size = int(.20 * len(population))

    contestants = sample(population, tournament_size)
    # Sort contestants based on their fitness
    contestants.sort(key=fitness, reverse=True)
   
    while len(parents) < 2 and count < 50: # ?
        count += 1

        for i in range(len(contestants)):
            cumulative_prob += p * (remaining_prob)
            remaining_prob *= (1 - p)

            if random() <= cumulative_prob and contestants[i] not in parents:
                parents.append(contestants[i])

                if len(parents) == 2:
                    break

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
def single_point_crossover(parent1, parent2, crossover_rate=0.7):
    """
    Single-point crossover between two abusive parents, splitting at a random
    point in the chromosome, where the the genes after that point will be
    swapped with the other parent.
    ----------------------------------------------------
    INPUT:
        parent1: (list) 
        parent2: (list)
        crossover_rate: (float; default=0.7) Probability of 

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

    # Sanity check ?

    return child1, child2

def k_point_crossover():
    pass

# Mutation
def bit_flip_mutation(child):
    """
    Mutation of child via BIT-FLIP, determined by the mutationrate:
    ~ 1 / len(child).
    ---------------------------------------------
    INPUT:
        child: (list)

    OUTPUT:
        mutated_child: (list)
    """
    mutation_rate = 1 / len(child)
    
    mutated_child = [1 - gene if random() < mutation_rate else gene for
                          gene in child]

    return mutated_child

def main():
    pass

population = get_population(knapsack_data["items"], 50)
lst = []
for i in range(5000):
    parent1, parent2 = tournament_selection(population)
    child1, child2 = single_point_crossover(parent1, parent2)
    mutant1, mutant2 = bit_flip_mutation(child1), bit_flip_mutation(child2)

breakpoint()
