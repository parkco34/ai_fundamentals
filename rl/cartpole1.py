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

    # [bin_size] * state_space -> creates list of bin_size repeated state_space
    # times, and w/ action dimension (+ [action_space]), becoming: [30, 30, 30, 30, 2]
    # initializes Q-table w/ random value tween -1 and 1
    q_table = np.random.uniform(low=-1, high=1, size=[bin_size] * state_space
                             + [action_space])

    return q_table, bins # q_table.ndim == 5, list of 4 np arrays

def discrete(state, bins):
    """
    Discretizes continuous space
    ----------------------------
    INPUT:
        state: (int) Continuous state space.
        bins: (list of ndarrays) discretized bins.

    OUTPUT:
        (tuple) for making it hashable for indexing into multidimensional
        (5-dimensional table, here: q_table.ndim) Q-table.
    """
    index = []

    for i in range(len(state)):
        # Corresponding Indices of bins; discretization process
        index.append(np.digitize(state[i], bins[i])-1)

    return tuple(index)

def q_learning(q_table, bins, episodes=5000, discount=0.95, alpha=0.1,
               timestep=100, epsilon=0.2):
    """
    Learns iteratively, the optimal Q-value function using the Bellman
    equation via storing Q-values in the Q-table to be updated @ each time
    step.
    ------------------------------------------------------------
    INPUT:
        q_table: (np.arrays) initialized Q-table
        bins: (list of np.arrays) Discretized bins for state variables 
        episodes: (int) Number of episodes to train over
        discount: (float) Discount factor determining importance of future
            rewards
        alpha: (float) Learning rate
        timestep: (int) Interfval reporting training progress
        epsilon: Initial exploration rate

    OUTPUT:
        (None)
    """
    # Initializations ?
    reward, steps = 0, 0

    for episode in range(1, episodes+1):
        steps += 1
        # Initialize indices for Q-table
        initial_state, _ = env.reset() # numpy array, empty dict
        current_state = discrete(initial_state, bins)

        score = 0
        done =  False

        for t in range(timestep):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # env.action_space -> All possible actions in environment
                # sample() -> randomly selects an action via (left=0, right=1)
                action = env.action_space.sample() # Explore w/ prob. 𝜀
                print(f"Explore!\nValue")

            else:
                # Returns index of highest Q-value
                # q_table[current_state] -> Array of Q-values for possible
                # actions.
                # np.argmax() -> finds index of max value in array
                action = np.argmax(q_table[current_state]) # Exploit; prob. 1-𝜀

            # take action, observing new state/reward
            # anv.step(action) -> (observation, reward, terminated, truncated, info)
            next_state, reward, done, _booly, _empty_dict= env.step(action)
            next_state = discrete(next_state, bins)
                        
            # UPdate Q-table
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + discount * q_table[next_state +
                                                    (best_next_action, )] # ?




q_table, bins = Qtable()

breakpoint()

#def main():
#    q_table, bins = Qtable()
#    
#    breakpoint()
#
#
#if __name__ == "__main__":
#    # To see all environments in gymnasium
##    gym.pprint_registry()
#    main()



