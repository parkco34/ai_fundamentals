#!/usr/bin/env pythonL
import numpy as np
import matplotlib.pyplot as plt



# Sample from two variables with mean zero, standard deviation one, and a given correlation coefficient
def get_samples(n_samples, correlation_coeff=0.8):
  a = np.random.normal(size=(1,n_samples))
  temp = np.random.normal(size=(1, n_samples))
  b = correlation_coeff * a + np.sqrt(1-correlation_coeff * correlation_coeff) * temp
  return a, b


N = 10000000
a,b, = get_samples(N)

# Verify that these two variables have zero mean and unit standard deviation
print("Mean of a = %3.3f,  Std of a = %3.3f"%(np.mean(a),np.std(a)))
print("Mean of b = %3.3f,  Std of b = %3.3f"%(np.mean(b),np.std(b)))

n_estimate = 1000000

N = 5

# TODO -- sample N examples of variable
# Compute the mean of each
# Compute the mean and variance of these estimates of the mean
# Replace this line
mean_of_estimator_1 = -1; std_of_estimator_1 = -1

print("Standard estimator mean = %3.3f, Standard estimator variance = \
      %3.3f"%(mean_of_estimator_1, std_of_estimator_1))


n_estimate = 1000000

N = 5

# TODO -- sample N examples of variables  and 
# Compute  for each and then compute the mean of 
# Compute the mean and variance of these estimates of the mean of 
# Replace this line
mean_of_estimator_2 = -1; std_of_estimator_2 = -1

print("Control variate estimator mean = %3.3f, Control variate estimator variance = %3.3f"%(mean_of_estimator_2, std_of_estimator_2))
