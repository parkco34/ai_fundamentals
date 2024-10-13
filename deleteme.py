#!/usr/bin/env python
import pandas as pd

# Define the data for the table
data_exercise_problem = {
    'Current State (s)': ['high', 'high', 'high', 'high', 'low', 'low', 'low', 'low', 'low'],
    'Action (a)': ['search', 'search', 'wait', 'wait', 'search', 'search', 'wait', 'wait', 'recharge'],
    'Next State (s\')': ['high', 'low', 'high', 'low', 'high', 'low', 'low', 'low', 'high'],
    'Reward (r)': [1, -3, 0, 0, -3, 'r_search', 0, 0, 0],
    'P(s\', r | s, a)': ['alpha', '1 - alpha', 1, 0, '1 - beta', 'beta', 1, 0, 1]
}

# Create the DataFrame
df_exercise_problem = pd.DataFrame(data_exercise_problem)

# Display the DataFrame in the console
print(df_exercise_problem)

# Optionally, save the DataFrame to a CSV file if you want to review it in another format
df_exercise_problem.to_csv("exercise_problem_table.csv", index=False)

