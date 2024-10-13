#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# Define the MDP components
states = [0, 1, 2, 3, 4]
actions = ['a', 'b']
gamma = 0.9  # Discount factor
lambda_factor = 0.5  # Weighting factor between forward and backward value functions

# Transition probabilities P(s'|s,a)
transition_probabilities = {
    0: {'a': {1: 1.0}, 'b': {2: 1.0}},
    1: {'a': {2: 1.0}, 'b': {3: 1.0}},
    2: {'a': {3: 1.0}, 'b': {4: 1.0}},
    3: {'a': {4: 1.0}, 'b': {0: 1.0}},
    4: {'a': {0: 1.0}, 'b': {1: 1.0}},
}

# Reward function R(s,a)
reward_function = {
    0: {'a': 1, 'b': 0},
    1: {'a': 0, 'b': 1},
    2: {'a': 1, 'b': 0},
    3: {'a': 0, 'b': 1},
    4: {'a': 1, 'b': 0},
}

# Initialize value functions
V_forward = np.zeros(len(states))
V_backward = np.zeros(len(states))
V_combined = np.zeros(len(states))

# Policy (random for simplicity)
policy = {
    state: {action: 0.5 for action in actions} for state in states
}

# Function to compute the forward value function
def compute_forward_value_function():
    V = np.zeros(len(states))
    threshold = 1e-6
    delta = float('inf')
    iteration = 0
    while delta > threshold:
        delta = 0
        for s in states:
            v = V[s]
            V_s = 0
            for a in actions:
                action_prob = policy[s][a]
                reward = reward_function[s][a]
                V_s_a = 0
                for s_prime in transition_probabilities[s][a]:
                    prob = transition_probabilities[s][a][s_prime]
                    V_s_a += prob * V[s_prime]
                V_s += action_prob * (reward + gamma * V_s_a)
            V[s] = V_s
            delta = max(delta, abs(v - V[s]))
        iteration += 1
    return V

# Function to compute the backward value function
def compute_backward_value_function():
    V = np.zeros(len(states))
    threshold = 1e-6
    delta = float('inf')
    iteration = 0
    # Reverse transition probabilities
    reverse_transition_probabilities = {s: {a: {} for a in actions} for s in states}
    for s in states:
        for a in actions:
            for s_prime in transition_probabilities[s][a]:
                prob = transition_probabilities[s][a][s_prime]
                if a in reverse_transition_probabilities[s_prime]:
                    if s in reverse_transition_probabilities[s_prime][a]:
                        reverse_transition_probabilities[s_prime][a][s] += prob
                    else:
                        reverse_transition_probabilities[s_prime][a][s] = prob
                else:
                    reverse_transition_probabilities[s_prime][a][s] = prob
    while delta > threshold:
        delta = 0
        for s in states:
            v = V[s]
            V_s = 0
            for a in actions:
                action_prob = policy[s][a]
                reward = reward_function[s][a]
                V_s_a = 0
                for s_prev in reverse_transition_probabilities[s][a]:
                    prob = reverse_transition_probabilities[s][a][s_prev]
                    V_s_a += prob * V[s_prev]
                V_s += action_prob * (reward + gamma * V_s_a)
            V[s] = V_s
            delta = max(delta, abs(v - V[s]))
        iteration += 1
    return V

# Compute value functions
V_forward = compute_forward_value_function()
V_backward = compute_backward_value_function()
V_combined = lambda_factor * V_forward + (1 - lambda_factor) * V_backward

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(states, V_forward, label='Forward Value Function', marker='o')
plt.plot(states, V_backward, label='Backward Value Function', marker='s')
plt.plot(states, V_combined, label='Combined Value Function', marker='^')
plt.title('Value Functions in Temporal Duality Reinforcement Learning')
plt.xlabel('States')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


