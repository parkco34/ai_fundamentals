#!/usr/bin/env python
from scipy.optimize import linprog

# Coefficients for the objective function (maximize profit)
c = [-35, -45, -65]  # Negative because linprog does minimization

# Coefficients for the constraints
A = [
    [3.5, 6, 8],   # Machine 1 hours constraint
    [4, 5, 6],     # Machine 2 hours constraint
    [11, 15, 20],  # Manpower constraint
    [-1, -1, 1]    # Policy constraint (x3 <= x1 + x2)
]

# Right-hand side values for the constraints
b = [120, 100, 280, 0]

# Bounds for the decision variables (x1, x2, x3)
x_bounds = (0, None)  # x1 >= 0
y_bounds = (0, None)  # x2 >= 0
z_bounds = (0, None)  # x3 >= 0

# Solving the linear programming problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds, z_bounds], method='highs')

# Extracting the solution and profit
x1, x2, x3 = result.x
max_profit = -result.fun

print(f"x1, x2, x3, max_profit: {x1}, {x2}, {x3}, {max_profit}")


