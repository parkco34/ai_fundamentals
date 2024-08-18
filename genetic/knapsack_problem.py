#!/usr/bin/env python

def roulette_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    relative_fitness = [fitness_scores[i] / total_fitness for i in
                        range(len(fitness_scores))]
    cumulative_probabilities = [sum(relative_fitness[:i+1] for i
                                    in range(len(relative_fitness)))]


