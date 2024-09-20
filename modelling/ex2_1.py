#!/usr/bin/env python
"""
### Problem Statement:

The Terre Haute Door Company (THDC) designs three types of steel doors: Standard, High Security, and Maximum Security. Each door requires different amounts of machine and labor time and has different profit margins; this information is given in the following table:

| Door Type          | Machine 1 Hours | Machine 1 Manpower | Machine 2 Hours | Machine 2 Manpower | Profit Margin |
|--------------------|-----------------|--------------------|-----------------|--------------------|---------------|
| **Standard**       | 3.5             | 5                  | 4               | 6                  | $35           |
| **High Security**  | 6               | 8                  | 5               | 7                  | $45           |
| **Maximum Security** | 8             | 11                 | 6               | 9                  | $65           |

Each door must go through both Machine 1 and Machine 2 before it can be sold. Each worker is assigned to work on only one of the doors, which means they work on both machines. In addition, management has decided not to sell more Maximum Security doors than the combined total of Standard and High Security doors, in order to keep demand high for Standard and High Security doors.

THDC has available to it only 120 hours per week on Machine 1 and 100 hours on Machine 2 before required maintenance, and 280 hours of manpower available per week. If we assume that we can sell every door that we make, how many of each door should be produced each week in order to maximize profits?
"""
import numpy as np

A = np.array([
    [3.5, 6, 8],
    [4, 5, 6],
    [11, 15, 20]
])

b = np.array([120, 100, 280])


breakpoint()

# ----------------------------------------------------
## Bounds for the decision variables (x1, x2, x3)
#x_bounds = (0, None)  # x1 >= 0
#y_bounds = (0, None)  # x2 >= 0
#z_bounds = (0, None)  # x3 >= 0
#
## Solving the linear programming problem
#result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds, z_bounds], method='highs')
#
## Extracting the solution and profit
#x1, x2, x3 = result.x
#max_profit = -result.fun
#
#print(f"x1, x2, x3, max_profit: {x1}, {x2}, {x3}, {max_profit}")
# ----------------------------------------------------


