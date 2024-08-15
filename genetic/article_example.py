#!/usr/bin/env python
from textwrap import dedent
from random import random, sample, choice, randint

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
    Perform Roulette Wheel Selection for either one or multiple selections,
    producing the parents.
    ---------------------------------------------
    INPUT:
        population: (list of lists)
        fitness_scores: (list)
        num_selections: (int; default: 2) Number of individuals to select.

    OUTPUT;
        selected: (list) Selected chromosomes
    """
    pass # ?
   
def tournament_selection(population, fitness_scores, tournament_size=3,
                         num_selections=2):
    """
    Selects most fit via competetion (Tournament), producing the parents.
    ---------------------------------------------------
    INPUT:
        population: (list of lists)
        fitness_scores: (list)
        tournament_size: (int; default=3)
        num_selections: (int; default: 2) Number of individuals to select.

    OUTPUT;
        sub_population: (list) Selected chromosomes
    """
    pass # ?

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
    # Ensure parents are same length
    if len(parent1) != len(parent2):
        raise ValueError(dedent("""
Parents must be thes same length!
                         """))
    # Crossover rate
    if random() > crossover_rate:
        return parent1[:], parent2[:] # if no crossover

    # Randomly determine the k value w/ limit being one less than the length of
    # parent chromsome
    k = randint(1, min(len(parent1)-1, len(parent1)-1))

    # Ensure proper inputs
    if k >= len(parent1) or k < 1:
        raise ValueError(dedent("""
k must be a positive integer less than the length of
parent chromosome
        """))

    # Unique crossover points in sorted order
    crossover_points = sorted(sample(range(1, len(parent1)), k))
    chromosome_length = len(parent1)
    # Initialize children
    child1, child2 = [], []
    use_parent1 = True

    start = 0
    for point in crossover_points + [chromosome_length]:
        
        if use_parent1:
            child1.extend(parent1[start:point])
            child2.extend(parent2[start:point])

        else:
            child1.extend(parent2[start:point])
            child2.extend(parent1[start:point])
        
        use_parent1 = not use_parent1
        start = point

    # Ensure children are different from parents and eachother
    if child1 == parent1 or child1 == parent2 or child1 == child2:
        # If child identical to parent or other child, flip bit
        flip_index = sample(range(chromosome_length), 1)[0]
        child1[flip_index] = 1 - child1[flip_index]

    if child2 == parent1 or child2 == parent2 or child2 == child1:
        flip_index = sample(range(chromosome_length), 1)[0]
        child2[flip_index] = 1 - child2[flip_index]

    return child1, child2
    
def bit_flip_mutation(chromosome):
    """
    Mutation on a chromosome via BIT-FLIP.
    ------------------------------------------------
    INPUT:
        chromosome: (np.array)

    OUTPUT:
        mutated_chromosome: (list)
    """
    # rate ~ 1/L, where L = length of chromosome
    mutation_rate = 1 / len(chromosome)
    # Flip bits if probability less than mutation rate
    mutated_chromosome = [1 - gene if random() < mutation_rate else gene for
                         gene in chromosome]

    return mutated_chromosome

def genetic_algorithm():
    """
    Implements genetic algorithm for 0-1 knapsack problem.
    -------------------------------------------------
    INPUT:
        config_file: (str) File path

    OUTPUT:
        best_solution, best_fitness, generation: (tuple: (np.array, float, int))  
    """
    # initialization
    population, capacity, items, generation, stop = get_initial_population(knapsack_data)

    best_solution = None
    best_fit = 0

    # Looping over generations
    for gen in range(stop):
        fitness_scores = [fitness(chromosome, items, capacity) for chromosome
                          in population]
        best_fit_indx = fitness_scores.index(max(fitness_scores))

        # Track best solutions
        if fitness_scores[best_fit_indx] > best_fit:
            best_fit = fitness_scores[best_fit_indx]
            best_solution = population[best_fit_indx]

        # Selection: Roulette vs Tournament
        parents = roulette_selection(population, fitness_scores)
#        parents = tournament_selection(population, fitness_scores) # ? Adjust!
        
        # Crossover operator
        kids = []
        for i in range(0, len(parents), 2):
            # Msut be pairs of parents
            if i+1 < len(parents):
                child1, child2 = single_point_crossover(parents[i], parents[i+1])
                kids.extend([child1, child2])

        # Mutation 
        mutated_kids = [bit_flip_mutation(child) for child in kids]

        population = mutated_kids

    return best_solution, best_fit, gen



# Example usage
# ------------------------------------------------------------------------
population, capacity, S, generation, stop = \
get_initial_population(knapsack_data)
fits = [fitness(population[i], S, capacity) for i in range(len(population))]
roulette = roulette_selection(population, fits, 2)
#parent1, parent2 = roulette[0], roulette[1] # ? fix: parents SAME
#tournament = tournament_selection(population, fits)
#parent1, parent2 = tournament[0], tournament[1]
#k = randint(1, min(len(parent1)-1, len(parent1)-1))
#child1, child2 = k_point_crossover(parent1, parent2, k)
# ------------------------------------------------------------------------
#thing = genetic_algorithm()
breakpoint()






