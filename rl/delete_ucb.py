#!/usr/bin/env python
import math

class UpperConfidenceBound:
    def __init__(self, n_actions, c):
        self.n_actions = n_actions  # Number of actions (arms)
        self.c = c  # Exploration factor
        self.counts = [0] * n_actions  # Number of times each action has been selected
        self.values = [0.0] * n_actions  # Estimated values (average rewards) of each action

    def select_action(self, t):
        """Selects an action using the UCB formula."""
        ucb_values = []

        for action in range(self.n_actions):
            if self.counts[action] == 0:
                # If the action has never been selected, select it to ensure exploration
                return action

            # Calculate the UCB value for the action
            average_reward = self.values[action]
            exploration_bonus = self.c * math.sqrt(math.log(t + 1) / self.counts[action])
            ucb_value = average_reward + exploration_bonus
            ucb_values.append(ucb_value)

        # Select the action with the maximum UCB value
        return ucb_values.index(max(ucb_values))

    def update(self, action, reward):
        """Updates the estimated value of the chosen action based on the received reward."""
        # Increment the count of the chosen action
        self.counts[action] += 1

        # Update the estimated value using incremental formula
        n = self.counts[action]
        value = self.values[action]
        new_value = value + (1 / n) * (reward - value)
        self.values[action] = new_value

    def run(self, rewards, num_rounds):
        """Runs the UCB algorithm for a given number of rounds and records total rewards."""
        total_reward = 0
        for t in range(num_rounds):
            action = self.select_action(t)
            reward = rewards[action][self.counts[action] % len(rewards[action])]  # Simulate the reward
            self.update(action, reward)
            total_reward += reward
            print(f"Round {t + 1}: Action {action}, Reward {reward}, Total Reward {total_reward}")
        return total_reward


# Define the rewards for each action (arms)
rewards = [
    [1, 1, 0, 1, 1],  # Action 0 rewards
    [0, 1, 0, 0, 1],  # Action 1 rewards
    [1, 0, 1, 1, 0]   # Action 2 rewards
]

# Initialize UCB with 3 actions and an exploration factor of 2
ucb = UpperConfidenceBound(n_actions=3, c=2)

# Run the UCB algorithm for 100 rounds
total_reward = ucb.run(rewards, num_rounds=100)
print(f"Total reward after 100 rounds: {total_reward}")

