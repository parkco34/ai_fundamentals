#!/usr/bin/env python
from textwrap import dedent
from random import random, sample, choice

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
    capacity = data["capacity"] # Max weight
    population = [data["chromosomes"][i]["bits"] for i in
                  range(len(data["chromosomes"]))]
    n = len(population[0]) # Number of items (genes)
    pop_size = len(population)
    generation = 0 # Initial generation

    # Set of items: (weights, values)
    S = [(item["weight"], item["value"]) for item in data["items"]]
    # Stopping criteria
    stop = 10 # No reason ? Dynamic programming to handle this automatically...

    return population, capacity, S, generation, stop

def dot_product(V, U):
    """
    Calculates dot product for the two lists.
    ----------------------------------------
    INPUT:
        V: (list)
        U: (list)

    OUTPUT:
        dot: (int) Dot product
    """
    if len(V) != len(U):
        return 0

    return sum(i[0] * i[1] for i in zip(V, U))
   
def fitness(chromosome, weight_value_pair, capacity):
    """
    Fitness function via DOT PRODUCT.
    ---------------------------------
    INPUT:
        chromosome: (list)
        weight_value_pair: (tuple) Weight/value pairs.
        capacity: (int) Max weight limit of knapsack.

    OUTPUT:
        profit: (float) Dot product of the two vectors: (chromosomes and
        values).
    """
    # Split weights and values
    weights = [pair[0] for pair in weight_value_pair]
    values = [pair[1] for pair in weight_value_pair]
    
    cumulative_weight = dot_product(chromosome, weights)
    #  Ensure capacity isn't exceeded
    if cumulative_weight <= capacity:
        profit = dot_product(chromosome, values)

    else:
        profit = 0 # ?

    return profit

def roulette_selection(population, fitness_scores, num_selections=2):
    """
    Perform Roulette Wheel Selection for either one or multiple selections.
    ---------------------------------------------
    INPUT:
        population: (list of lists)
        fitness_scores: (list)
        num_selections: (int; default: 2) Number of individuals to select.

    OUTPUT;
        sub_population: (list) Selected chromosomes
    """
    sub_populations = []

    total_fitness = sum(fitness_scores)
    selection_probabilities = [fit / total_fitness for fit in fitness_scores]
    # "Spinning wheela" via random number generation contained within
    # probability distribution
    cumulative_probs = []
    cumulative_sum = 0

    # Getting cumulative probabilites which allows the selection process to
    # maintain exact probabilities assigned to individuals via their fitness
    for prob in selection_probabilities:
        cumulative_sum += prob
        cumulative_probs.append(cumulative_sum)
   
    # Selecting individuals in proportion to their fitness
    for peepz in range(num_selections):
        r = random()
        
        # Get sub-population that contains uniform random #
        for i, cumulative_prob in enumerate(cumulative_probs):
            if r <= cumulative_prob:
                sub_populations.append(population[i])
                break

    # Determine if single or multiple individuals to be returned
    if num_selections == 1:
        return sub_populations[0]

    else:
        return sub_populations

def tournament_selection(population, fitness_scores, tournament_size=3,
                         num_selections=2):
    """
    Selects most fit via competetion (Tournament).
    ---------------------------------------------------
    INPUT:
        population: (list of lists)
        fitness_scores: (list)
        tournament_size: (int; default=3)
        num_selections: (int; default: 2) Number of individuals to select.

    OUTPUT;
        sub_population: (list) Selected chromosomes
    """
    selected =[]

    for dude in range(num_selections):
        tournament_indices = sample(range(len(population)), tournament_size)
        tournament_dudes = [population[i] for i in tournament_indices]
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Highest fitness individual
        winner_index = tournament_fitness.index(max(tournament_fitness))
        winner = tournament_dudes[winner_index]

        selected.append(winner)

    # Number of dudes to return
    if num_selections == 1:
        return selected[0]

    else:
        return selected

def single_point_crossover(parent1, parent2, crossover_rate=0.7):
    """
    Single-point crossover between two abusive parents, splitting at a randomly
    selected point in chromosome for recombination.
    --------------------------------------------------
    INPUT:
        parent1: (list) Parent 1 chromosome.
        parent2: (list) Parent 2 chromosome.
        crossover_rate: (float; default=0.7) Probability of performing
        crossover ?

    OUTPUT:
        bastards: (tuple) Two offspring chromosomes.
    """
    # Crossover point, where I've chosent the interval such that there's no
    # chance of accidentally swapping the entire chromosome of one parent with
    # the other, leaving them unchanged.
    crossover_point = randint(1, len(parent1)-2)

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

def k_point_crossover(parent1, parent2, crossover_rate=0.7):
    """
    Combines chromosomes of parents to create offspring, inhereting best
    traits, using k-point crossover, which randomly selects k point in
    chromosome to split and recombine for offspring.
    ----------------------------------------
    INPUT:
        parent1: (list)
        parent2: (list)
        crossover_rate: (float; default=0.7)

    OUTPUT:
        bastards: (tuple) ?
    """
    # Randomly determine the k value: (1 <= k < len(parent1))
    k = randint(1, len(parent1)-1)
    # Initialize children
    child1, child2 = [], []

    # Ensure proper inputs
    if k >= len(parent1) or k < 1:
        raise ValueError(dedent("""
k must be a positive integer less than the length of
parent chromosome
        """))

    # Unique crossover points in sorted order
    crossover_points = sorted(sample(range(1, len(parent1)), k))

    # Crossover operation
    for i in range(len(parent1)):
        # Swap the kth position between parents
        if i in crossover_points:
            child1.append(parent2[i])
            child2.append(parent1[i])

        else:
            child1.append(parent1[i])
            child2.append(parent2[i])

    # Inidicate whether crossover operation did anything
    if child1 == parent1 and child2 == parent2:
        raise Exception("Crossover did nothing!")

    return child1, child2

def mutation(chromosome):
    """
    Mutation on a chromosome.
    ------------------------------------------------
    INPUT:
        chromosome: (np.array)
        mutation_rate: (float) Probability of mutating each gene.

    OUTPUT:
        mutated_chromosome: (np.array)
    """
    pass

def genetic_algorithm(config_file):
    """
    Implements genetic algorithm for 0-1 knapsack problem.
    -------------------------------------------------
    INPUT:
        config_file: (str) File path

    OUTPUT:
        best_solution, best_fitness, generation: (tuple: (np.array, float, int))  
    """
    pass


# Example usage
population, capacity, S, generation, stop = \
get_initial_population(knapsack_data)
fits = [fitness(population[i], S, capacity) for i in range(len(population))]
roulette = roulette_selection(population, fits, 2)
parent1, parent2 = roulette[0], roulette[1]
#tournament = tournament_selection(population, fits)
#parent1, parent2 = tournament[0], tournament[1]
breakpoint()

child1, child2 = k_point_crossover(parent1, parent2)







