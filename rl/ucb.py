#!/usr/bin/env python
import math

def initialize(n_actions, c):
    """
    Initialize:
        - number of actions
        - c, the exploration factor
    --------------------------------
    INPUT:

    OUTPUT:
    """
    pass

def selec_action(t):
    """
    Uses UCB formula to select actions via:
        - Select action if never been selected, returning it.
        - Calculate avg reward for action
        - ucb_value = avg reward + exploration bonus (c*sqrt(log(t+1)))
        - Store values
        - Select argmax of stored values
    """
    pass

def update(action, reward):
    """
    Update rule for estimated chosen action value based on the received reward.
    """
    pass

def run(rewards, num_rounds):
    """
    UPdate estimated values of chosen action based on recieved reward.
    """
    pass



