#!/usr/bin/env python
"""
Helped by:
    https://github.com/maciejbalawejder/Reinforcement-Learning-Collection/blob/main/Q-Table/Qtable.ipynb
    https://gymnasium.farama.org/environments/classic_control/cart_pole/

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
-----------------------------------------------------------
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
from tabulate import tabulate
import csv
from textwrap import dedent
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
# PART 2
import torch
import torch.nn as nn
import torch.optim as optim

# Example values
EXAMPLE_EPSILON = 1.0
EXAMPLE_EPSILON_MIN = 0.05
EXAMPLE_EPSILON_DECAY = 0.995
EXAMPLE_DISCOUNT=0.9
EXAMPLE_ALPHA=0.1
EXAMPLE_TIMESTEP = 100
EXAMPLE_BIN_SIZE = 30
EXAMPLE_EPISODES = 5000
EXAMPLE_EPSILON_MIN = 0.01
EXAMPLE_NUM_SAMPLES = 5000
EXAMPLE_MAX_STEPS = 500
# PART 2
EXAMPLE_HIDDEN_UNITS = 64

# Set random seeds for reproducibility
np.random.seed(73)

# CartPole environment
env = gym.make("CartPole-v1")
env.reset(seed=73)  # Reproducibility
# Outputs the lower/upper bounds of observation space
print(env.observation_space.low, "\n", env.observation_space.high)  # |S|,|A(s)|

def Qtable(state_space=len(env.observation_space.low),
           action_space=env.action_space.n, bin_size=EXAMPLE_BIN_SIZE):
    """
    Qtable: defined by number of states and actions...
    Increasing the number of bins increases memory space needed exponentially and takes
    longer to run, whereas too few bins results in sacrificing performance.
    ----------------------------------------------------------------
    Each element Q[i,j,k,l,m] represents the Q-value for:
        Cart position in bin i
        Cart velocity in bin j
        Pole angle in bin k
        Pole angular velocity in bin l
        Action m (0 or 1, corresponding to left or right)
    ----------------------------------------------------------------
    BINS: List of numpy arrays representing the discretized boundaries for one
    of the state variables in env., a numpy array for each state.
    ----------------------------------------------------------------
    INPUT:
        state_space: (int); default: 4 -> {cart_pos, cart_vel, pole_pos,
            pole_angle_vel}
        action_space: (int); default: 2 -> {right, left}
        bin_size: (int); default: 30

    OUTPUT:
        q_table, bins: (tuple of ndarrays ?, list of ndarrays)
    """
    # Ranges for state variables
    bins = [
        # Cart position
        np.linspace(-4.8, 4.8, bin_size),
        # Carrt velocity
        np.linspace(-3, 3, bin_size),
        # Pole position
        np.linspace(-0.418, 0.418, bin_size),
        # Pole velocity
        np.linspace(-4, 4, bin_size)
    ]

    # Initialize Q-table with zeros
    # [bin_size] * state_space + [action_space] == [30]*4 + [2] == [30, 30, 30,
    # 30, 2]
    q_table = np.zeros([bin_size] * state_space + [action_space])

    return q_table, bins

def discretize(state, bins):
    """
    DISCRETIZES CONTINUOUS SPACE by mapping each continuous state to a
    discrete bin index.
    Loops thru each each state variable, clipping state value within bin range
    to ensure the state value falls within range.
    HOW ================================================================
        min(state[i], bins[i][-1]): This ensures that state[i] does not exceed the maximum bin edge. 
        If state[i] is greater than bins[i][-1] (the last bin edge), it is set to bins[i][-1].
        max(..., bins[i][0]): This ensures that state[i] is not less than the minimum bin edge. 
        If state[i] is less than bins[i][0] (the first bin edge), it is set to bins[i][0].
    ================================================================
    ------------------------------------------------------------------
    INPUT:
        state: (array) Continuous state space.
        bins: (list of ndarrays) discretized bins.

    OUTPUT:
        indices: (tuple) for making it hashable for indexing into multidimensional
        (5-dimensional table, here: q_table.ndim) Q-table.
    """
    indices = []

    for i in range(len(state)):
        # Max(min(state_value, last bin interval), first bin interval)
        # Ensuring state values are within bins range
        value = max(min(state[i], bins[i][-1]), bins[i][0])
        # Corresponding indices of bins; discretization process
        # Adjusts index to start at 0 by subtracting 1
        indices.append(np.digitize(value, bins[i]) - 1)

    return tuple(indices)

def initialize_policy_and_value(bin_size=EXAMPLE_BIN_SIZE,
                                state_space=len(env.observation_space.low),
                                action_space=env.action_space.n):
    """
    Initial Policy and Value Function.
    --------------------------------------------------
    INPUT:
        bin_size: (int) Num of bins
        state_space: (int)
        action_space: (int)

    OUTPUT:
        policy, value_function: (np.ndarray, np.dnarray)
    """
    policy_shape = [bin_size] * state_space
    # Initialize randomly
#    policy = np.random.choice(action_space, size=policy_shape)
    # Initialize with zeros
    policy = np.random.choice(action_space, size=policy_shape)

    # Initialize value function
    value_function = np.zeros(policy_shape)

    return policy, value_function

def policy_evaluation(
    policy, 
    bins, 
    value_func, 
    discount=EXAMPLE_DISCOUNT, 
    alpha=EXAMPLE_ALPHA, 
    episodes=EXAMPLE_EPISODES):
    """
    Policy Evaluation.
    --------------------------------------------------
    INPUT:
        policy: (np.ndarray)
        bins: (list of np.ndarrays)
        value_func: (np.ndarray)
        episodes: (int)

    OUTPUT:
        value_func: (np.ndarray)
    """
    for episode in range(episodes):
        continuous_state, _ = env.reset()
        current_state = discretize(continuous_state, bins)
        done = False

        while not done:
            # ? Incorrect way to assign action
            action = policy[current_state]
            continuous_next_state, reward, done, booly, empty_dict = \
                env.step(action)
            next_state = discretize(continuous_next_state, bins)

            # Value function update: TD(0) update
            value_func[current_state] += alpha * (
                reward +
                discount * value_func[next_state] - value_func[current_state]
            )

            # Update current state
            current_state = next_state

            # Check for failed episode
            if done and reward == 0:
                value_func[current_state] = -1

    return value_func

def policy_improvement():
    """

    """
    pass

def q_learning(
    q_table,
    bins,
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    alpha=EXAMPLE_ALPHA,
    timestep=EXAMPLE_TIMESTEP,
    epsilon=EXAMPLE_EPSILON,
    epsilon_decay=EXAMPLE_EPSILON_DECAY,
    epsilon_min=EXAMPLE_EPSILON_MIN,
    performance_window=100, # Num of episodes to consider for performance
    epsilon_increase=0.1 # Amount to increase epsilon when performance drops
):
    """
    Learns iteratively, the optimal Q-value function using the Bellman equation
    via storing Q-values in the Q-table to be updated @ each timestep.
    ------------------
    ADAPTIVE EPSILON:
    ------------------
        -> Everytime timestep episodes (every EXAMPLE_TIMESTEP episode), the
        function checks if recent average reward over the last
        performance_window episodes is less than avg reward of previous
        performance_window episodes.
        
        -> If performance drop is detected, epsion increased by
        epsilon_increase (max=1.0 to ensure valid probabaility).
    --------------------------------------------------------
    INPUT:
        q_table: (np.arrays) initialized Q-table
        bins: (list of np.arrays) Discretized bins for state variables
        episodes: (int) Number of episodes to train over
        discount: (float) Discount factor determining importance of future
            rewards
        alpha: (float) Learning rate
        timestep: (int) Interval reporting training progress
        epsilon: Initial exploration rate
        epsilon_decay: (float) Rate at which epsilon decays
        epsilon_min: (float) Minimum value of epsilon

    OUTPUT:
        episode_rewards: (list) Total reward per episode
        q_table: (np.array) Updated Q-table
    """
    episode_rewards = []
    recent_avg_rewards = []
    
    for episode in range(episodes):
        # Iniitialize indices for Q-table
        continuous_state, _ = env.reset()
        current_state = discretize(continuous_state, bins)

        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection for EXPLORATION
            if np.random.random() < epsilon:
                action = env.action_space.sample()

            else:
                # EXPLOITATION via choosing action max Q-value
                action = np.argmax(q_table[current_state])

            # Execute chosen action
            next_continuous_state, reward, done, booly, empty_dict = env.step(action)
            next_state = discretize(next_continuous_state, bins)

            # Find action with highest Q-value in next state
            best_next_action = np.argmax(q_table[next_state])
            # TD target
            td_target = reward + discount * q_table[next_state + (best_next_action, )]
            td_error = td_target - q_table[current_state + (action, )]
            # Update Q-value
            q_table[current_state + (action, )] += alpha * td_error

            # Update current state and total reward
            current_state = next_state
            total_reward += reward

        # Epsilon decay to reduce exploration over time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        else:
            epsilon = epsilon_min

        episode_rewards.append(total_reward)

        # Adaptive Epsilon IMplementation
        # track performance
        if episode % timestep == 0 and episode >= performance_window:
            recent_avg_reward = np.mean(episode_rewards[-performance_window:])
            previous_avg_reward = \
            np.mean(episode_rewards[-2*performance_window:-performance_window])

            # If performance drops, increase epsilon slightly
            if recent_avg_reward < previous_avg_reward:
                epsilon = min(epsilon + epsilon_increase, 1.0)
                print(f"Performance dropped. Increasing epsilon to {epsilon:.4f}")
                
        # Output progress at every timestep
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"Q-Learning -> Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()

    return episode_rewards, q_table


# Example implementation
q_table, bins = Qtable()
policy, value_func = initialize_policy_and_value()
thing = policy_evaluation(policy, bins, value_func)



#breakpoint()
