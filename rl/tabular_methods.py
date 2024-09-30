#!/usr/bin/env python
"""
TABULAR METHODS W/ DISCRETIZATION
Convert continuos state space into finite set of discrete states.
"""
import numpy as np

# Number of bins for each variable
num_bins = [10, 10, 10, 10]

# Define bins for each state variable
cart_pos_bins = np.linspace(-4.8, 4.8, num_bins[0] + 1)[1:-1]
cart_vel_bins = np.linspace(-3.0, 3.0, num_bins[1] + 1)[1:-1]
# -24°, +24° in radians
pole_angle_bins = np.linspace(-0.418, 0.418, num_bins[2] + 1)[1:-1]
pole_vel_bins = np.linspace(-3.0, 3.0, num_bins[3] + 1)[1:-1]

# Mapping functino
def discrete_state(observation):
    cart_pos, cart_vel, pole_angle, pole_vel = observation

    # Discretize each variable
    cart_pos_disc = np.digitize(cart_pos, cart_pos_bins)
    cart_vel_disc = np.digitize(cart_vel, cart_vel_bins)
    pole_angle_disc = np.digitize(pole_angle, pole_angle_bins)
    pole_vel_disc = np.digitize(pole_vel, pole_vel_bins)

    # Combine discretized variables into a single state
    state = (cart_pos_disc, cart_vel_disc, pole_angle_disc, poel_vel_disc)

    return state




breakpoint()

