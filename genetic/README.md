---

# Genetic Algorithm for 0-1 Knapsack Problem

## Overview

This project implements a Genetic Algorithm (GA) to solve the 0-1 Knapsack problem. The problem involves selecting items, each with a specific weight and value, to include in a knapsack such that the total weight does not exceed a given capacity, while maximizing the total value.

## Prerequisites

- Python 3.x
- NumPy

## File Descriptions

- `config_1.txt`: Configuration file for the first problem instance. Contains parameters for the GA and item data.
- `config_2.txt`: Configuration file for the second problem instance. Contains parameters for the GA and item data.
- `knapsack_ga.py`: Python script that implements the Genetic Algorithm.

## Steps to Solve the Problem

### 1. **Representation and Initialization**
   - The GA starts by randomly initializing a population of individuals, each represented as a binary chromosome string where `1` means an item is included in the knapsack, and `0` means it is not included.
   - The initialization includes the population size, number of generations, knapsack capacity, and item list, which are read from the configuration file.

### 2. **Fitness Function**
   - The fitness of each chromosome is calculated based on the total value of the included items.
   - The fitness is set to zero if the total weight of the included items exceeds the knapsack's capacity.

### 3. **Selection**
   - Two selection methods are implemented: **Roulette Wheel Selection** and **Tournament Selection**.
   - These methods are used to select parent chromosomes for crossover based on their fitness.

### 4. **Crossover**
   - A single-point crossover method is implemented. The crossover point is chosen randomly, and the offspring inherit genes from both parents.

### 5. **Mutation**
   - Genes in the chromosome may mutate with a probability `Mr` within a specified range. Mutation involves flipping the bit (0 to 1 or 1 to 0).

### 6. **Stop Criterion**
   - The algorithm stops after a pre-defined number of generations (`stop`), and the best solution in the final generation is returned as the output.

### 7. **Extra Credit Options**
   - Implement custom fitness functions or stop criteria to explore different approaches and compare their effectiveness.

## Running the Code

1. Clone the repository or download the code.
2. Ensure that the `config_1.txt` and `config_2.txt` files are in the same directory as the script.
3. Run the `knapsack_ga.py` script:
   ```bash
   python knapsack_ga.py config_1.txt
   ```
4. The script will output the best solution found by the GA, including the items selected and their total value.

## Configuration File Format

The configuration files `config_1.txt` and `config_2.txt` contain the following data:

1. Population size (`pop_size`)
2. Number of generations (`n`)
3. Mutation rate (`Mr`)
4. Knapsack capacity (`W`)
5. Item list where each item is represented as a tuple `(weight, value)`

Example format:
```
50      # pop_size
100     # n
45      # Mr
2341    # W
208 2   # Item list starts here
...
```

## Experimentation

- To explore different configurations, modify the `config_1.txt` or `config_2.txt` files with different values for `pop_size`, `n`, `Mr`, `W`, and item list.
- Run multiple experiments and compare results to analyze the impact of different GA parameters.

## Reporting

- For each configuration, plot the fitness of the population at each generation.
- Compare the performance of Roulette Wheel Selection vs. Tournament Selection.
- Analyze the number of active genes and fitness for the best solution throughout the experiment.

## Conclusion

This project demonstrates the application of Genetic Algorithms to a classical optimization problem. By tweaking various parameters and selection methods, different aspects of the GA's performance can be observed and analyzed.

--- 

This `README.md` should guide you through setting up, running, and analyzing the Genetic Algorithm for the 0-1 Knapsack problem using the provided configuration files.
