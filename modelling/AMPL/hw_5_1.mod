##############
# Model File
##############
set J;
set P{j in J};

param tmin{j in J};
param tmax{j in J};
param cmax{j in J};
param cmin{j in J};
param T;

var s{j in J} >= 0;
var t{j in J} >= tmin[j], <= tmax[j];
var c{j in J} >= cmin[j], <= cmax[j];

minimize Cost: sum{j in J} c[j];
s.t. LinearCost{j in J} : 
c[j] = (t[j] - tmin[j]) * (cmin[j] - cmax[j]) / (tmax[j] - tmin[j]) + cmax[j];
s.t. Precedence{j in J, i in P[j]} : s[j] >= s[i] + t[i];
s.t. Deadline{j in J} : s[j] + t[j] <= T;
