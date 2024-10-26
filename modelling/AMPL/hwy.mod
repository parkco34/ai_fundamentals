# Incorrect code:
# ---------------
#set J;
#
#param u{j in J};
#param r{j in J};
#param M; # Total num of officers
#
# D.V.s
#var x{j in J} >= 0, <= u[j];
# Cannot have variable in bound expression !
#var z{j in J} <= 0, integer; # Number of policeman assigned
#
#maximize MinReductions: sum{j in J} z[j];
#s.t. MinConstraint{j in J} : z[j] <= r[j] * x[j];
#s.t. TotalOfficers : sum{j in J} x[j] = M;
#====================================================
#The objective is to maximize the minimum reduction zz. 
# So, we want zz to be as large as possible, constrained by the 
# actual reduction r[j]×x[j]r[j]×x[j] on each segment.
# If we write the constraint as z≤r[j]×x[j]z≤r[j]×x[j], 
# it allows zz to be any small value as 
# long as it satisfies the inequality. 
# Since we want to push zz upwards, this does not work in ourfavor.
#====================================================

# CORRECT -----------------------------
set J;

# Parameters
param r{j in J}; # Potential speeding reduction per officer assigned to segment j
param u{j in J}; # Maximum number of officers that can be assigned to patrol segment j
param M;         # Total number of police officers

# Decision Variables
var z >= 0;          # Minimum speeding reduction across any segment
var x{j in J} >= 0, <= u[j], integer; # Number of officers assigned to each segment

# Objective: Maximize the minimum speeding reduction across all segments
maximize MinSpeedReduction: z;

# Constraints:
s.t. MinReductionConstraint{j in J}: r[j] * x[j] >= z; # Ensure that the reduction on each segment is at least z
s.t. OfficerLimitConstraint{j in J}: x[j] <= u[j];     # Ensure we don't assign more officers than the upper limit
s.t. TotalOfficers: sum{j in J} x[j] = M;              # The total number of officers assigned must equal M

