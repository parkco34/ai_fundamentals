#!/usr/bin/env python
import pulp
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

def solve_diet_problem(F, N, c_f, a_n_f, m_n, M_n, available_F):
    """
    Solves the diet problem for a given set of available food items.

    Parameters:
    - F: List of all food items.
    - N: List of all nutritional requirements.
    - c_f: Dictionary mapping food items to their cost.
    - a_n_f: Dictionary of dictionaries mapping nutrition to food items and their amounts.
    - m_n: Dictionary mapping nutrition to minimum required amounts.
    - M_n: Dictionary mapping nutrition to maximum allowed amounts.
    - available_F: Set of food items available for selection.

    Returns:
    - status: Status of the optimization ('Optimal', etc.).
    - meal: Dictionary mapping food items to the amount selected.
    - total_cost: Total cost of the meal.
    """
    # Define the problem
    prob = LpProblem("Diet_Problem", LpMinimize)

    # Define decision variables
    X = LpVariable.dicts("Food", available_F, lowBound=0, cat='Continuous')

    # Objective function: minimize total cost
    prob += lpSum([c_f[f] * X[f] for f in available_F]), "Total_Cost"

    # Constraints:
    for n in N:
        # Minimum requirement
        prob += lpSum([a_n_f[n][f] * X[f] for f in available_F]) >= m_n[n], f"Min_{n}"
        # Maximum requirement
        prob += lpSum([a_n_f[n][f] * X[f] for f in available_F]) <= M_n[n], f"Max_{n}"

    # Solve the problem
    prob.solve()

    # Check if an optimal solution was found
    if LpStatus[prob.status] != 'Optimal':
        return LpStatus[prob.status], None, None

    # Retrieve the solution
    meal = {f: X[f].varValue for f in available_F if X[f].varValue > 1e-5}
    total_cost = value(prob.objective)

    return LpStatus[prob.status], meal, total_cost

def main():
    # Sample Data
    F = ['Chicken', 'Beef', 'Fish', 'Rice', 'Beans', 'Broccoli', 'Carrots', 'Milk', 'Eggs']
    N = ['Calories', 'Protein', 'Fat', 'Carbohydrates', 'Vitamins']

    # Cost per unit (e.g., per gram or appropriate unit)
    c_f = {
        'Chicken': 2.0,
        'Beef': 3.0,
        'Fish': 2.5,
        'Rice': 1.0,
        'Beans': 1.2,
        'Broccoli': 1.5,
        'Carrots': 1.3,
        'Milk': 0.8,
        'Eggs': 0.5
    }

    # Nutritional content per unit
    a_n_f = {
        'Calories': {
            'Chicken': 250,
            'Beef': 300,
            'Fish': 200,
            'Rice': 130,
            'Beans': 120,
            'Broccoli': 50,
            'Carrots': 40,
            'Milk': 100,
            'Eggs': 150
        },
        'Protein': {
            'Chicken': 30,
            'Beef': 25,
            'Fish': 20,
            'Rice': 2.5,
            'Beans': 15,
            'Broccoli': 4,
            'Carrots': 1,
            'Milk': 8,
            'Eggs': 12
        },
        'Fat': {
            'Chicken': 5,
            'Beef': 15,
            'Fish': 7,
            'Rice': 0.5,
            'Beans': 1,
            'Broccoli': 0.5,
            'Carrots': 0.2,
            'Milk': 5,
            'Eggs': 10
        },
        'Carbohydrates': {
            'Chicken': 0,
            'Beef': 0,
            'Fish': 0,
            'Rice': 28,
            'Beans': 22,
            'Broccoli': 10,
            'Carrots': 9,
            'Milk': 12,
            'Eggs': 1
        },
        'Vitamins': {
            'Chicken': 5,
            'Beef': 4,
            'Fish': 6,
            'Rice': 2,
            'Beans': 3,
            'Broccoli': 8,
            'Carrots': 7,
            'Milk': 6,
            'Eggs': 5
        }
    }

    # Minimum nutritional requirements
    m_n = {
        'Calories': 2000,
        'Protein': 50,
        'Fat': 70,
        'Carbohydrates': 300,
        'Vitamins': 100
    }

    # Maximum nutritional limits
    M_n = {
        'Calories': 2500,
        'Protein': 200,
        'Fat': 100,
        'Carbohydrates': 350,
        'Vitamins': 150
    }

    # Number of days to plan
    D = 5

    # Initialize available food items
    available_F = set(F)

    # To store meals for each day
    meals = []

    for day in range(1, D + 1):
        status, meal, total_cost = solve_diet_problem(F, N, c_f, a_n_f, m_n, M_n, available_F)

        if status != 'Optimal':
            print(f"Day {day}: No optimal solution found. Status: {status}")
            break

        print(f"\nDay {day}:")
        print("Meal Composition:")
        for food, amount in meal.items():
            print(f"  {food}: {amount:.2f} units")
        print(f"Total Cost: ${total_cost:.2f}")

        meals.append(meal)

        # Heuristic to ensure diversity: Remove one food item used in today's meal
        # You can choose different strategies, such as removing multiple items or based on certain criteria
        # Here, we'll remove the most expensive food item used in the meal
        if meal:
            # Find the food item with the highest cost in the current meal
            most_expensive_food = max(meal, key=lambda f: c_f[f])
            available_F.remove(most_expensive_food)
            print(f"  -> Excluding '{most_expensive_food}' from available food items for the next day.")

    # Optional: Reintroduce food items after some days to allow their reuse
    # For example, after all days are planned, reset the available_F
    # available_F = set(F)

if __name__ == "__main__":
    main()
