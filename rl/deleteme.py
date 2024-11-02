#!/usr/bin/env python
import numpy as np

class UCBAgent:
    def __init__(self, n_actions, c=1.0):
        """
        Initialize the UCB agent.

        Parameters:
        - n_actions (int): Number of actions (arms).
        - c (float): Exploration parameter. Higher values of `c` encourage more exploration.
        """
        self.n_actions = n_actions
        self.c = c
        self.counts = np.zeros(n_actions)  # Counts of how many times each action was chosen
        self.values = np.zeros(n_actions)  # Estimated reward for each action
        self.total_steps = 0               # Total steps taken (t in the equation)

    def select_action(self):
        """
        Select an action using the Upper-Confidence-Bound (UCB) formula.

        Returns:
        - action (int): Index of the chosen action.
        """
        # Increment the total steps counter
        self.total_steps += 1

        # Compute the UCB value for each action
        ucb_values = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            if self.counts[action] == 0:
                # If action hasn't been chosen, prioritize it to encourage exploration
                ucb_values[action] = float('inf')
            else:
                # UCB formula
                avg_reward = self.values[action]
                confidence_bound = self.c * np.sqrt(np.log(self.total_steps) / self.counts[action])
                ucb_values[action] = avg_reward + confidence_bound

        # Select the action with the maximum UCB value
        action = np.argmax(ucb_values)
        return action

    def update(self, action, reward):
        """
        Update the action-value estimates based on the received reward.

        Parameters:
        - action (int): The action that was chosen.
        - reward (float): The reward received for choosing that action.
        """
        # Update the counts for the chosen action
        self.counts[action] += 1

        # Update the estimated value (average reward) for the chosen action
        n = self.counts[action]
        current_value = self.values[action]

        # Incremental formula for updating the average
        new_value = current_value + (reward - current_value) / n
        self.values[action] = new_value

# Example usage
if __name__ == "__main__":
    n_actions = 10  # Assume we have 10 actions (arms)
    agent = UCBAgent(n_actions, c=2.0)

    # Simulate a simple environment
    true_rewards = np.random.rand(n_actions)  # Random true reward probabilities for each action

    for _ in range(1000):  # Run for 1000 steps
        action = agent.select_action()
        # Simulate a reward based on the true reward probability of the chosen action
        reward = np.random.binomial(1, true_rewards[action])
        agent.update(action, reward)

    print("Estimated values:", agent.values)
    print("Action counts:", agent.counts)


