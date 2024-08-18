#!/usr/bin/env python
"""
Sources used for learning:
    - https://faun.pub/genetic-algorithms-to-solve-the-zero-one-knapsack-problem-implementation-26c1982f44b3
    - 
"""
from textwrap import dedent
from random import random, sample, choice, randint, uniform

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

def input_validation(population, fitness_scores, **kwargs):
    """
    Validates inputs for genetic algorithm functions.
    ------------------------------------------------------
    INPUT:
        population: (list) Individuals
        fitness_scores: (list)
        **kwargs: Additional parameters to validate:
            - tournament_size: (int)
            - num_selections: (int)
            - crossover_rate: (float)
            - mutation_rate: (float)

    OUTPUT:
        None: Raises ValueError if inputs are invalid
    """
    # Ensure proper data dypes
    if not isinstance(population, list) or not isinstance(fitness_scores,
                                                          list):
        raise ValueError("Population and fitness scores must be lists")

    # Fitnesses much corrspond to individual in population
    if len(population) != len(fitness_scores):
        raise ValueError("Population and fitness scores must be same length")

    if len(population) == 0:
        raise ValueError("Population cannot be empty")

    if any(score < 0 for score in fitness_scores):
        raise ValueError("Fitness scores cannot be negative")

    # For additional parameters for other functions
    if "tournament_size" in kwargs:
        tournament_size = kwargs["tournament_size"]

        if not isinstance(tournament_size, int) or (tournament_size < 1) or ( 
        tournament_size > len(population)):
            raise ValueError("""Tournament size must be an integer between 1 and
                             the population size""")
    if "num_selections" in kwargs:
        num_selections = kwargs["num_selections"]

        if not isinstance(num_selections, int) or num_selections < 1:
            raise ValueError("Number of selections must be a positive integer")

    if "crossover_rate" in kwargs:
        crossover_rate = kwargs["crossover_rate"]

        if not isinstance(crossover_rate, (int, float)) or not (0 <= crossover_rate <=
                                                         1):
            raise ValueError("""Crossover rate needs to be a float/int between 0 and
                             1""")
    if "mutation_rate" in kwargs:
        mutation_rate = kwargs["mutation_rate"]

        if not isinstance(mutation_rate, (int, float)) or not (0 <=
                                                               mutation_rate <=
                                                               1):
            raise ValueError("""Mutation rate needs to be a floating point number
                             , or integer, between 0 and 1""")

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
        profit = 1e-6

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
        parents: (list) parent chromosomes
    """
    # INput validations
    input_validation(population, fitness_scores, num_selections=num_selections)

    # individuals for seleciton
    selected = []

    # Total fitness of population
    total_fit = sum(fitness_scores)

    # Total fitness of zero results in random sampling
    if total_fit == 0:
        return sample(population, num_selections)

    # Relative fitness of individuals
    relative_fits = [fit/total_fit for fit in fitness_scores]
    # cumulative probabiities for "segments" fo the roulette wheel, where the
    # final cumulative probability is 1 for the whole wheel
    cumulative_probabilities = [sum(relative_fits[:i+1]) for i in range(len(relative_fits))]
     
    # Initialization
    selected = []
    attempts = 0
    # Max number of attempts to avoid infinite loop
    max_attempts = 100
    
    # infinite loop here! ?
    while len(selected) < num_selections:
        # Uniform random number for the "spin" landing on point
        point = uniform(0,1)
        # "Spinning" the wheel
        for i, prob in enumerate(cumulative_probabilities):
            # Wheel stops spinning 
            if point <= prob:
                # Ensure unique parents
                if population[i] not in selected:
                    selected.append(population[i])

                break

        attempts += 1
        if attempts > max_attempts:
            print("""Max attempts reached; adding random individual to avoid
                  infinite loop""")
            remaining = [ind for ind in population if ind not in selected]

            if not remaining:
                print("Error: No remaining individuals to select from")
                break

            selected.append(choice(remaining))
            break

    return selected
    
def tournament_selection(population, fitness_scores, tournament_size=3,
                         num_selections=2):
    """
    Selects most fit via competetion (Tournament), producing the parents by
    randomly choosing the tournament size K from the population, pooling them,
    obtaining the most fit from the pool, producing the parents for
    reproduction.
    ---------------------------------------------------
    INPUT:
        population: (list of lists)
        fitness_scores: (list)
        tournament_size: (int; default=3)
        num_selections: (int; default: 2) Number of individuals to select.

    OUTPUT;
        selected: (list) Selected chromosomes
    """
    # input validation
    input_validation(population, fitness_scores,
                     tournament_size=tournament_size,
                     num_selections=num_selections)

    # Ensure number of selected individuals doesn't exceed population
    if num_selections > len(population):
        raise ValueError("Number of selections shouldn't exceed population")

    selected = [] # Use set, instead... ?

    while len(selected) < num_selections:
        # Random selection of contestants
        contestants = sample(population, tournament_size)
        contestant_fitness = [fitness_scores[population.index(c)] for c in
                                             contestants]
        # index of best contestant
        winner_index = contestant_fitness.index(max(contestant_fitness))
        winner = contestants[winner_index]

        # Ensure parent isn't asexual (>▽<)
        if winner not in selected:
            selected.append(contestants[winner_index])

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
    # Mutation rate ~ 1/L, where L = length of chromosome
    mutation_rate = 1 / len(chromosome)
    # Flip bits if probability less than mutation rate
    mutated_chromosome = [1 - gene if random() < mutation_rate else gene for
                         gene in chromosome]

    return mutated_chromosome

def genetic_algorithm(selection_function=roulette_selection,
                      crossover_function=single_point_crossover,
                      mutation_function=bit_flip_mutation, **kwargs):
    """
    Implements genetic algorithm for 0-1 knapsack problem.
    -------------------------------------------------
    INPUT:
        func: (func; default: roulette_selection) Selection function
        **kwargs: Additional parameters to validate:
            - tournament_size: (int)
            - num_selections: (int)
            - crossover_rate: (float)
            - mutation_rate: (float)

    OUTPUT:
        best_solution, best_fitness, generation: (tuple: (np.array, float, int))  
    """
    # Dictionary to store values for comparison
    things = {"population":[], "fitness": [], 
              "parents": [], "children": [], "best_fit": [], "generation": []}

    # initialization
    population, capacity, items, generation, stop = get_initial_population(knapsack_data)
    fitness_scores = [fitness(chromosome, items, capacity) for chromosome in population]
    
    best_soution = None
    best_fitness = float("-inf")

    # Main loop
    while generation < stop:
        things["population"].append(population)
        things["fitness"].append(fitness_scores)
        # Selection: Initially just the original population
        parents = selection_function(population, fitness_scores,
                       num_selections=len(population), **kwargs)
        # Crossover
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = crossover_function(parent1, parent2, **kwargs)
            next_generation.extend([child1, child2])

        # Mutation
        next_generation = [mutation_function(child) for child in
                           next_generation]

        # Evaluate new generation's fitness
        fitness_scores = [fitness(chromosome, items, capacity) for chromosome
                           in next_generation]

        # Update population
        population = next_generation

        # Track best solution
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[fitness_scores.index(best_fitness)]

        # Store values in dictionary for debugging
        things["parents"].append(parents)
        things["best_fit"].append(best_fitness)
        things["generation"].append(generation)

        # New generation
        generation += 1

    print(f"Dictionary for debugging: {thing}")
    return best_solution, best_fitness, generation

# Infinite loop somewhere ... ?


# Example usage
# ------------------------------------------------------------------------
population, capacity, S, generation, stop = \
get_initial_population(knapsack_data)
fits = [fitness(population[i], S, capacity) for i in range(len(population))]
#tournament = tournament_selection(population, fits)
#roulette = roulette_selection(population, fits)
#parent1, parent2 = roulette[0], roulette[1] # ? fix: parents SAME
#tournament = tournament_selection(population, fits)
#parent1, parent2 = tournament[0], tournament[1]
#k = randint(1, min(len(parent1)-1, len(parent1)-1))
#child1, child2 = k_point_crossover(parent1, parent2, k)
# ------------------------------------------------------------------------
#thing = genetic_algorithm()
next_generation = []
parents = roulette_selection(population, fits, num_selections=len(population))

breakpoint()

# ?
"""
1. What if child same as parent
2. 
"""
