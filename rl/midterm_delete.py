#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Part 1: Grid World MDP Analysis
def calculate_value(state: Tuple[int, int], next_state: Tuple[int, int], grid: Dict) -> float:
    """Calculate reward for a transition"""
    if next_state in grid:
        if grid[next_state] == 'G':
            return 1.0
        elif grid[next_state] == 'D':
            return -1.0
    return 0.0

def get_next_state(state: Tuple[int, int], action: str) -> Tuple[int, int]:
    """Get next state given current state and action"""
    x, y = state
    if action == 'UP':
        return (x-1, y) if x > 0 else (x, y)
    elif action == 'DOWN':
        return (x+1, y) if x < 1 else (x, y)
    elif action == 'LEFT':
        return (x, y-1) if y > 0 else (x, y)
    else:  # RIGHT
        return (x, y+1) if y < 1 else (x, y)

# Define grid
grid = {
    (0,0): 'S',
    (0,1): 'D',
    (1,0): '0',
    (1,1): 'G'
}

# Policy evaluation for π₀ (always go right)
gamma = 0.9
V = {state: 0.0 for state in grid.keys()}

# One iteration of policy evaluation
for state in grid.keys():
    if grid[state] != 'G':  # Goal state value remains 0
        next_state = get_next_state(state, 'RIGHT')
        reward = calculate_value(state, next_state, grid)
        V[state] = reward + gamma * V[next_state]

print("Value function after one iteration:")
for i in range(2):
    print(f"|{V[(i,0)]:6.3f}|{V[(i,1)]:6.3f}|")

# Policy improvement
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
Q = {}
new_policy = {}

for state in grid.keys():
    if grid[state] != 'G':
        Q[state] = {}
        for action in actions:
            next_state = get_next_state(state, action)
            reward = calculate_value(state, next_state, grid)
            Q[state][action] = reward + gamma * V[next_state]
        new_policy[state] = max(Q[state].items(), key=lambda x: x[1])[0]

print("\nQ-values for each state-action pair:")
for state in Q:
    print(f"\nState {grid[state]} at {state}:")
    for action, value in Q[state].items():
        print(f"{action}: {value:.3f}")

print("\nNew policy π₁:")
for i in range(2):
    row = []
    for j in range(2):
        if (i,j) in new_policy:
            row.append(new_policy[(i,j)])
        else:
            row.append('·')
    print(f"|{row[0]:^5}|{row[1]:^5}|")

# Part 2: Multi-Armed Bandit Implementation

class BanditAlgorithm:
    def __init__(self, n_arms: int, true_means: List[float]):
        self.n_arms = n_arms
        self.true_means = true_means
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.optimal_arm = np.argmax(true_means)

    def pull_arm(self, arm: int) -> float:
        return np.random.normal(self.true_means[arm], 1)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, n_arms: int, true_means: List[float], epsilon: float):
        super().__init__(n_arms, true_means)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)

class UCB(BanditAlgorithm):
    def __init__(self, n_arms: int, true_means: List[float]):
        super().__init__(n_arms, true_means)
        self.total_pulls = 0

    def select_arm(self) -> int:
        self.total_pulls += 1
        # Initialize all arms if needed
        if np.any(self.counts == 0):
            return np.where(self.counts == 0)[0][0]

        ucb_values = self.values + np.sqrt(2 * np.log(self.total_pulls) / self.counts)
        return np.argmax(ucb_values)

# Run simulation
np.random.seed(42)
n_arms = 5
true_means = [0.2, 0.5, 0.8, 0.1, 0.4]
n_steps = 1000
n_runs = 100

def run_simulation(algorithm_class, **kwargs):
    cumulative_rewards = np.zeros((n_runs, n_steps))
    regrets = np.zeros((n_runs, n_steps))
    optimal_reward = max(true_means)

    for run in range(n_runs):
        algorithm = algorithm_class(n_arms, true_means, **kwargs)

        for step in range(n_steps):
            arm = algorithm.select_arm()
            reward = algorithm.pull_arm(arm)
            algorithm.update(arm, reward)

            cumulative_rewards[run, step] = reward
            regrets[run, step] = optimal_reward - true_means[arm]

    return cumulative_rewards, regrets

# Run both algorithms
eps_rewards, eps_regrets = run_simulation(EpsilonGreedy, epsilon=0.1)
ucb_rewards, ucb_regrets = run_simulation(UCB)

# Plot results
plt.figure(figsize=(12, 5))

# Cumulative Rewards
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(np.mean(eps_rewards, axis=0)), label='ε-Greedy')
plt.plot(np.cumsum(np.mean(ucb_rewards, axis=0)), label='UCB')
plt.xlabel('Time steps')
plt.ylabel('Cumulative Reward')
plt.title('Average Cumulative Reward')
plt.legend()

# Cumulative Regret
plt.subplot(1, 2, 2)
plt.plot(np.cumsum(np.mean(eps_regrets, axis=0)), label='ε-Greedy')
plt.plot(np.cumsum(np.mean(ucb_regrets, axis=0)), label='UCB')
plt.xlabel('Time steps')
plt.ylabel('Cumulative Regret')
plt.title('Average Cumulative Regret')
plt.legend()

plt.tight_layout()
plt.show()

