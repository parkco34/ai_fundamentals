#!/usr/bin/env python
"""
--------------------
CARTPOLE-V0 PROBLEM:
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
--------------------
The classic CartPole-v0 problem from OpenAI Gym. This environment presents a continuous state space and a discrete action space, offering an excellent platform to compare tabular methods with function ap- proximation techniques.
------------------------------------------------------------
Learning Objectives
• Implement and understand policy iteration, temporal difference (TD) learning, and Monte Carlo (MC) methods.
• Apply discretization techniques to handle continuous state spaces in tabular methods.
• Utilize function approximation to manage high-dimensional or continuous state spaces.
• Compare the performance of different RL algorithms and understand their advantages and limitations.
------------------------------------------------------------
Background
The CartPole-v0 environment consists of a pole attached by an unactuated joint to a cart, which moves along a frictionless track. The goal is to prevent the pole from falling over by applying forces (left or right) to the cart. The environment provides four continuous observations:
1. Cart Position
2. Cart Velocity
3. Pole Angle
4. Pole Velocity at Tip
------------------------------------------------------------
================
Assignment Tasks
================
PART 1: 
Tabular Methods with Discretization
-------------------------
1. State Space Discretization
Objective: Convert the continuous state space into a finite set of discrete states. Instructions:
• Define bins for each of the four state variables.
• Create a mapping function to convert continuous observations into discrete states.

2. Policy Iteration
OBJECTIVE: Implement policy evaluation and improvement to find the optimal policy. Instructions:
• Perform iterative policy evaluation until the value function converges.
3. Temporal Difference Learning (SARSA and Q-Learning)
Objective: Implement online learning algorithms to estimate the action-value function. Instructions:
• Implement SARSA and Q-Learning algorithms using the discretized state space.
4. Monte Carlo Methods
Objective: Use episodic sampling to estimate the value of states.
Instructions:
• Implement Monte Carlo prediction.
5. Performance Evaluation
Objective: Compare the performance of the tabular methods.
Instructions:
• Define evaluation metrics (e.g., average return per episode, number of episodes to convergence). Plot learning curves for each method.
• Analyze the stability and efficiency of the algorithms.

PART 2:

Function Approximation
-------------------------
1. Linear Function Approximation
Objective: Implement TD learning with linear function approximation.
Instructions:
• Represent the Q-function as a linear combination of features.
2. Non-Linear Function Approximation (Neural Networks)
Objective: Implement TD learning with neural networks.
Instructions:
• Design a neural network to approximate the Q-function.
• Decide on the architecture (number of layers, neurons, activation functions).
3. Monte Carlo Methods with Function Approximation
Objective: Combine MC methods with function approximation.
Instructions:
• Use the returns from episodes to update the function approximator.
• Ensure that the function approximator generalizes well across states.
4. Performance Evaluation
Objective: Compare the performance of function approximation methods with tabular methods.
Instructions:
• Use the same evaluation metrics as before.
• Discuss the impact of function approximation on learning speed and policy quality.
• Analyze the effects of hyperparameters and network architecture.
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# CartPole environment
env = gym.make("CartPole-v1")
# Outputs the lower/upper bounds of observation space
print(env.observation_space.low, "\n", env.observation_space.high) # |S|,|A(s)|

def Qtable(state_space=len(env.observation_space.low), action_space=env.action_space.n, bin_size=30):
    """
    Q-TABLE: Defined by the number of state and actions...
    ------------------------------------------------------
    Defines bins with normalization for extreme values velocity values.
    ------------------------------------------------
    INPUT:
        state_space: (int); default: 4 -> {cart_pos, cart_vel, pole_pos,
            pole_angle_vel}
        action_space: (int); default: 2 -> {right, left}
        bin_size: (int); default: 30

    OUTPUT:
        q_table, bins: (tuple of ndarrays ?, list of ndarryas)
    """
    # Ranges for each of the state variables, where they're evenly spaced
    # intervals
    bins = [
        # Cart position (in meters)
        np.linspace(-4.8, 4.8, bin_size),
        # Cart velocity
        np.linspace(-4, 4, bin_size),
        # Pole position
        np.linspace(-0.418, 0.418, bin_size),
        # Pole angular velocity (in radians/second)
        np.linspace(-4, 4, bin_size)
    ]

    """ Creating the Q-table, where its size established via [bin_size] *
    state_space + [action_space] resulting in CartPole shape: 
      [30, 30, 30, 30, 2] """
    q_table = np.random.uniform(low=-1, high=1, size=[bin_size] * state_space
                             + [action_space])

    return q_table, bins

def discrete(state, bins):
    """
    Discretizes continuous space
    ----------------------------
    INPUT:
        state: (int) Continuous state space.
        bins: (list of ndarrays) discretized bins.

    OUTPUT:
        (tuple) for making it hashable for indexing into multidimensional
        (4-dimensional table, here) Q-table.
    """
    index = []

    for i in range(len(state)):
        # Boundaries on state values
        state_value = np.clip(state[i], bins[i][0], bins[i][-1])
        # Corresponding Indices of bins
        index.append(np.digitize(state_value, bins[i])-1)

    return tuple(index)

def q_learning(q_table, bins, episodes=5000, discount=0.9, alpha=0.2,
               timestep=5000, epsilon=0.8, epsilon_decay=0.99,
               epsilon_min=0.01):
    """
    ?
    """
    pass

def main():
    q_table = Qtable()



if __name__ == "__main__":
    # To see all environments in gymnasium
#    gym.pprint_registry()
    main()



