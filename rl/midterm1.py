#!/usr/bin/env python
"""
Cory Parker
Midterm
"""
import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy(true_means, epsilon=0.1, timesteps=1000, experiments=100):
    """
    Epsilon-greedy algorithm implementation.
    ?
    ---------------------------------------
    INPUT:

    OUTPUT:
    """
    # Number of arms
    arms = len(true_means)
    rewards = np.zeros((experiments, timesteps))

    for trial in range(experiments):
        # Estimated value of each arm
        Q = np.zeros(arms)
        # Number of times each arm selected
        N = np.zeros(arms)

        for t in range(timesteps):
            # Exploration vs. Exploitation
            if np.random.rand() < epsilon:
                # Explore
                action = np.random.randint(arms)

            else:
                # Exploit
                action = np.argmax(Q)

            # Reward from arm with variance = 1
            reward = np.random.normal(true_means[action], 1)

            # Updates
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]

            # Record reward obtained
            rewards[trial, t] = reward

    # Average all reward over all experiments for each time step
    avg_rewards = np.mean(rewards, axis=0)

    return avg_rewards

def ucb(true_means, explore=2, timesteps=1000, experiments=100):
    """
    Upper-Confidence-Bound Action Selection.
    ----------------------------------------
    INPUT:

    OUTPUT:
    """
    arms = len(true_means)
    rewards = np.zeros((experiments, timesteps))

    for trial in range(experiments):
        Q = np.zeros(arms)
        N = np.zeros(arms)

        for t in range(timesteps):
            if t < arms:
                # Initializze each
                action = t

            else:
                # UCB for each arm and prevent division by zero in denominator
                ucb_values = Q + explore * np.sqrt(np.log(t) / (N + 1e-5))
                action = np.argmax(ucb_values)

            # Reward from selected arm
            reward = np.random.normal(true_means[action], 1)

            # Updates
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]

            # Record reward
            rewards[trial, t] = reward

    # Avg rewards over all
    avg_rewards = np.mean(rewards, axis=0)

    return avg_rewards

def avg_regrets(true_means, avg_rewards_egreedy, avg_rewards_ucb,
                    timesteps=1000):
    """
    Returns the Average Regret for epsilon-greedy and UCB after number of
    timesteps.
    --------------------------------------------------
    INPUT:
        true_means: (np.ndarray)
        avg_rewards_egreedy: ()
        avg_rewards_ucb: ()
        timesteps: (int)

    OUTPUT:
        egreedy_regret: (float) Total regret for e-greedy 
        ucb_regret: (float) 
    """
    # Best possible mean reward
    optimal_mean = np.max(true_means)

    # Cumulative optimal rewards
    optimal_cum_reward = optimal_mean * np.arange(1, timesteps+1)

    # Cumulative Rewards
    cum_egreedy_reward = np.cumsum(avg_rewards_egreedy)
    cum_ucb_reward = np.cumsum(avg_rewards_ucb)

    # Regret - difference between the optimal cumulative reward and cumulative
    # reward
    regret_egreedy = optimal_cum_reward[-1] - cum_egreedy_reward[-1]
    regret_ucb = optimal_cum_reward[-1] - cum_ucb_reward[-1]

    print(f"Average Regret after {timesteps} time steps:")
    print(f"e-greedy {regret_egreedy:.2f}" )
    print(f"UCB: {regret_ucb:.2f}")

    return regret_egreedy, regret_ucb

def plot_avg_rewards(avg_rewards_egreedy, avg_rewards_ucb):
    """
    Visualization.
    --------------------------------------
    INPUT:
        avg_rewards_greedy: (np.nadarray)
        avg_rewards_ucb: (np.ndarray)

    OUTPUT:
        None
    """
    # Cumulative sum of rewards
    rewards_greedy = np.cumsum(avg_rewards_egreedy)
    rewards_ucb = np.cumsum(avg_rewards_ucb)

    plt.figure(figsize=(12, 8))
    plt.plot(rewards_greedy, label="epsilon-greedy (epsilon=0.1)")
    plt.plot(rewards_ucb, label="UPPER CONFIDENCE BOUND (UCB)")

    # Labels
    plt.title("Average Cumulative Reward over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")

    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    np.random.seed(73)

    # For 5 arms
    true_means = np.random.normal(0,1, 5)
    # Epsilon-greedy run
    avg_greedy_rewards = epsilon_greedy(true_means)
    # Upper confidence bound action selection
    avg_rewards_ucb = ucb(true_means)

    # Plot
    plot_avg_rewards(avg_greedy_rewards, avg_rewards_ucb)
    avg_regrets(true_means, avg_greedy_rewards, avg_rewards_ucb)


if __name__ == "__main__":
    main()
