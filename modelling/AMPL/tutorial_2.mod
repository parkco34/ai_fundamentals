# Sets
set MILLS;   # Set of mills (Gary, Cleveland, Pittsburgh)
set PLANTS;  # Set of plants (Framingham, Detroit, etc.)
set PRODUCTS; # Set of products (bars, coils, plates)

# Parameters
param HoursPerTon {MILLS, PRODUCTS};  # Hours required per ton for each product at each mill
param Cost {MILLS, PRODUCTS};         # Cost per ton for each product at each mill
param Demand {PLANTS, PRODUCTS};      # Demand (tons) of each product at each plant
param AvailableHours {MILLS};         # Available working hours each

# Decision variables
var X {MILLS, PLANTS, PRODUCTS} >= 0; # Tons shipped from mills to plants for each product

# Objective function: Minimize total cost
minimize TotalCost:
    sum {m in MILLS, p in PLANTS, t in PRODUCTS} Cost[m,t] * X[m,p,t];

# Constraints
# Demand satisfaction
subject to Demand_Satisfaction {p in PLANTS, t in PRODUCTS}:
    sum {m in MILLS} X[m,p,t] = Demand[p,t];

# Working hours limitation
subject to Hours_Limit {m in MILLS}:
    sum {p in PLANTS, t in PRODUCTS} HoursPerTon[m,t] * X[m,p,t] <= AvailableHours[m];
