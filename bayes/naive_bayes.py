#!/usr/bin/env python
from math import exp, sqrt, pi

def read_data(filename):
    """
    Reads data.
    ---------------
    INPUT:
        filename: (str) Aboslute or Relative file path 

    OUTPUT:
        dataset: (list)
    """
    # Ensure user enters appropriate filename.
    if not filename:
        print(f"You did enter a valid filename!")

    with open(filename ,"r") as file:
        text = file.read()

    return text

def split_data(dataset):
    """
    Splits data into training and test data.
    -----------------------------------
    INPUT:
        dataset: (dict) ?

    OUTPUT:
        features, classes: (tuple of ?)
    """
    pass

def mean(num):
    return sum(num) / len(num)

def summarize_data_classes(features, classes):
    """
    Summarizes dataset via calcilating mean/variance per feature/class.
    ------------------------------
    INPUT:
        features: (?)
        classes: (?)

    OUTPUT:
        summaries: (dict)
    """
    pass

breakpoint()
