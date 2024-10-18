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
import numpy as np
import matplotlib.pyplot as plt
import time

# Example values
EXAMPLE_EPSILON = 1.0
EXAMPLE_EPSILON_MIN = 0.01
EXAMPLE_EPSILON_DECAY = 0.995
EXAMPLE_DISCOUNT=0.9
EXAMPLE_ALPHA=0.1
EXAMPLE_TIMESTEP = 100
EXAMPLE_BIN_SIZE = 30
EXAMPLE_EPISODES = 5000
EXAMPLE_EPSILON_MIN = 0.01


# Set random seeds for reproducibility
np.random.seed(73)

# CartPole environment
env = gym.make("CartPole-v1")
env.reset(seed=73)  # Reproducibility
# Outputs the lower/upper bounds of observation space
print(env.observation_space.low, "\n", env.observation_space.high)  # |S|,|A(s)|

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
        q_table, bins: (tuple of ndarrays ?, list of ndarrays)
    """
    # Adjusted ranges for the state variables
    bins = [
        # Cart position (bounded between -4.8 and 4.8)
        np.linspace(-4.8, 4.8, bin_size),
        # Cart velocity (limited manually)
        np.linspace(-3, 3, bin_size),
        # Pole angle (bounded between -0.418 and 0.418 radians)
        np.linspace(-0.418, 0.418, bin_size),
        # Pole angular velocity (limited manually)
        np.linspace(-4, 4, bin_size)
    ]

    # Initialize Q-table with zeros
    q_table = np.zeros([bin_size] * state_space + [action_space])

    return q_table, bins  # q_table.ndim == 5, list of 4 np arrays

def discrete(state, bins):
    """
    Discretizes continuous space
    ----------------------------
    INPUT:
        state: (array) Continuous state space.
        bins: (list of ndarrays) discretized bins.

    OUTPUT:
        indices: (tuple) for making it hashable for indexing into multidimensional
        (5-dimensional table, here: q_table.ndim) Q-table.
    """
    index = []

    for i in range(len(state)):
        # Clip the state values to be within the bins range
        value = max(min(state[i], bins[i][-1]), bins[i][0])
        # Corresponding Indices of bins; discretization process
        index.append(np.digitize(value, bins[i]) - 1)

    indices = tuple(index)

    return indices

def q_learning(
    q_table, 
    bins, 
    episodes=EXAMPLE_EPISODES,
    discount=EXAMPLE_DISCOUNT,
    alpha=EXAMPLE_ALPHA,
    timestep=EXAMPLE_TIMESTEP, epsilon=EXAMPLE_EPSILON,
    epsilon_decay=EXAMPLE_EPSILON_DECAY,
    epsilon_min=EXAMPLE_EPSILON_MIN
):
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
        timestep: (int) Interval reporting training progress
        epsilon: Initial exploration rate
        epsilon_decay: (float) Rate at which epsilon decays
        epsilon_min: (float) Minimum value of epsilon

    OUTPUT:
        episode_rewards: (list) Total reward per episode
        q_table: (np.array) Updated Q-table
    """
    episode_rewards = []

    for episode in range(1, episodes + 1):
        # Initialize indices for Q-table
        state_continuous, _ = env.reset()
        current_state = discrete(state_continuous, bins)

        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else:
                # Exploit: choose the action with max Q-value
                action = np.argmax(q_table[current_state])

            # Execute chosen action
            next_state_continuous, reward, done, _, _ = env.step(action)
            # Discretize new continuous state for indexing Q-table
            next_state = discrete(next_state_continuous, bins)
            # Find action with highest Q-value in next state
            best_next_action = np.argmax(q_table[next_state])
            # TD target
            td_target = reward + discount * q_table[next_state + (best_next_action,)]
            # TD error
            td_error = td_target - q_table[current_state + (action,)]
            # Update Q-value
            q_table[current_state + (action,)] += alpha * td_error

            # Update current state and total reward
            current_state = next_state
            total_reward += reward

        # Decay epsilon to reduce exploration over time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_min

        episode_rewards.append(total_reward)

        # Output progress at every timestep
        if episode % timestep == 0:
            avg_reward = np.mean(episode_rewards[-timestep:])
            print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    env.close()

    return episode_rewards, q_table

def initialize_policy(state_space=len(env.observation_space.low),
                      action_space=env.action_space.n,
                      bin_size=EXAMPLE_BIN_SIZE):
    """
    Initializes Policy, randomly.
    --------------------------------
    INPUT:
        state_space: (int)
        action_space: (int)
        bin_size: (int)

    OUTPUT:
        policy: (ndarray) Policy table mapping each discrete state to action.
         - Multi-dimensional array where each element represents a state,
                              and the value is a randomly chosen action for that state.
                              The shape of this array is [bin_size, bin_size, bin_size, bin_size],
                              one dimension for each state variable.
    """
    policy = np.random.choice(action_space, size=[bin_size] * state_space)

    return policy

def policy_evaluation(policy, bins, value_func, discount=EXAMPLE_DISCOUNT,
                      alpha=EXAMPLE_ALPHA, episodes=EXAMPLE_EPISODES):
    """
    Policy Evaluation: Computing V_𝞹(s) for arbitrary 𝞹.
    ----------------------------------------------------
    INPUT:
        policy: (ndarray) Current policy mapping state to actions
        bins: (list of ndarrays) Discretized bins for state variables
        value_func: (ndarray) Current value function to be updated
        discount: (float) Discount factor
        alpha: (float) Learning rate
        episodes:(int) Number of episodes for evaluation

    OUTPUT:
        value_func: (ndarray) Updated value function
    """
    for episode in range(episodes):
        continuous_state, _ = env.reset()
        current_state = discrete(continuous_state, bins)
        done = False

        while not done:
            action = policy[current_state]
            next_state_continuous, reward, done, booly, empty_dict = \
            env.step(action)
            
            # Update value function using TD(0) update
            value_func[current_state] += alpha * (reward + discount *
                                                  value_func[next_state] -
                                                  value_func[current_state])

            curren_state = next_state

    return value_func
        

def sarsa(env, episodes, alpha, discount, epsilon):
    pass

def monte_carlo(env, episodes, discount):
    pass

def performance_evalutation(rewaeds, algorithm_name):
    pass

def main():
    pass    

#    results = {
#        "Policy Iteration":,
#        "SARSA": ,
#        "Q-Learning": ,
#        "Monte Carlo":
#    }

q_table, bins = Qtable()

breakpoint()
