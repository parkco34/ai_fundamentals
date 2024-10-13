#!/usr/bin/env python
"""
--------------------
CartPole-V0 Problem:
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








