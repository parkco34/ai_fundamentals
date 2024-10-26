#!/usr/bin/env python
from sympy import symbols, solve, simplify

# Define symbolic variables
t_j, c_j = symbols('t_j c_j')  # Current time and cost
t_min, t_max = symbols('t_j^min t_j^max')  # Min and max time
c_min, c_max = symbols('c_j^min c_j^max')  # Min and max cost

def derive_time_cost_relationship():
    # Step 1: Calculate slope using the two points
    # Point 1: (t_min, c_max)
    # Point 2: (t_max, c_min)
    slope = (c_min - c_max)/(t_max - t_min)
    print("Slope (m_j):")
    print(f"m_j = {slope}\n")

    # Step 2: Use point-slope form of line equation
    # c_j - c_max = m(t_j - t_min)
    equation = c_j - c_max - slope*(t_j - t_min)
    print("Initial equation using point-slope form:")
    print(f"{equation} = 0\n")

    # Step 3: Solve for c_j
    solution = solve(equation, c_j)[0]
    print("Solved for c_j:")
    print(f"c_j = {solution}\n")

    # Step 4: Simplify by factoring out negative sign
    # Replace (c_min - c_max) with -(c_max - c_min)
    simplified = solution.subs(c_min - c_max, -(c_max - c_min))
    print("Final simplified equation:")
    print(f"c_j = {simplified}\n")

    return simplified

def verify_solution(equation):
    # Verify the solution works for the endpoints

    # Test Case 1: When t_j = t_min, should get c_j = c_max
    result_min = simplify(equation.subs(t_j, t_min))
    print("Verification at minimum time:")
    print(f"When t_j = t_min:")
    print(f"c_j = {result_min}")
    print(f"Equals c_max? {simplify(result_min - c_max) == 0}\n")

    # Test Case 2: When t_j = t_max, should get c_j = c_min
    result_max = simplify(equation.subs(t_j, t_max))
    print("Verification at maximum time:")
    print(f"When t_j = t_max:")
    print(f"c_j = {result_max}")
    print(f"Equals c_min? {simplify(result_max - c_min) == 0}\n")

def example_calculation():
    # Example with numeric values
    equation = derive_time_cost_relationship()

    # Substitute example values
    t_min_val, t_max_val = 4, 7
    c_min_val, c_max_val = 5000, 8000
    t_j_val = 5

    numeric_result = equation.subs([
        (t_j, t_j_val),
        (t_min, t_min_val),
        (t_max, t_max_val),
        (c_min, c_min_val),
        (c_max, c_max_val)
    ])

    print("Example calculation:")
    print(f"For t_j = {t_j_val}, t_min = {t_min_val}, t_max = {t_max_val}")
    print(f"c_min = ${c_min_val}, c_max = ${c_max_val}")
    print(f"Result: c_j = ${numeric_result}")

if __name__ == "__main__":
    print("Deriving Time-Cost Relationship:\n")
    equation = derive_time_cost_relationship()
    verify_solution(equation)
    example_calculation()

