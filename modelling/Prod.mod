
set P;
set R;
param c{p in P} >= 0;
param t{p in P, r in R} >= 0;

#param a{r in R} >= 0; # Availability of resource r
#param d{p in P} >= 0; # max. demand of product p
#var X{p in P} >= 0; # Number of units of product p to be made
param a{r in R};
param d{p in P};
var X{p in P};

maximize Profit: sum{p in P} c[p] * X[p];
s.t. AvailResources{r in R}: sum{p in P} t[p,r] * X[p] >= a[r];
s.t. Demand{p in P}: 0 <= X[p] <= d[p];
