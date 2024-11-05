#!/usr/bin/env python
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum

setup_cost = {
    'DET': 20000,
    'ATL': 18000,
    'SA': 10000,
    'DEN': 5000,
    'CIN': 6000,
    'NO': 5000
}

shipping_cost = {
    'DET': {'DEN': 1253, 'CIN': 637, 'NO': 1128},
    'ATL': {'DEN': 1398, 'CIN': 841, 'NO': 702},
    'SA': {'DEN': 942, 'CIN': 1154, 'NO': 691},
    'NO': {'LA': 1983, 'CHI': 1272, 'PHI': 1751}
}

capacity = {
    'DET': 150,
    'ATL': 150,
    'SA': 120
}

demand = {
    'LA': 80,
    'CHI': 100,
    'PHI': 70
}

prob = LpProblem("Plant_Warehouse_Optimization", LpMinimize)
# Defines binary decision variables
y = {i: LpVariable(f"y_{i}", cat=Binary) for i in setup_costs.keys()}
x = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous") for i in
     shipping_costs.keys() for j in shipping_costs[i].keys()}




