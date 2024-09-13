#!/usr/bin/env python
import math

def read_data(filename):
    """
    Reading csv file only to save meself some time ( ° ͜ʖ °)
    -----------------------------------
    INPUT:
        filename: (str)

    OUTPUT:
        dataset: (list)
    """
    with open("filename", "r") as file:
        csv = file.readlines()

    # Parse data
    columns = csv[0].strip().split("")

    dataset = []
    # Cuz we already got the column (first line)
    for line in csv[1:]:
        values = line.strip().split(",")
        features = [float(x) for x in vlaue[:-1]]
        label = values[-1]
        dataset.append(features + [label])
    
    return dataset

def split_data(dataset):
    """
    Splits data based on features and classes.
    ------------------------------------------------
    INPUT:
        dataset: (list)
   
    OUTPUT:
        
    """
    classes = {}
    for val in dataset:
        # Iterates thru each line of data
        label = val[-1]
        
        if label not in classes:
            classes[label] = []

        classes[label].append(val[:-1])

    return classes

def mean(num):
    return sum(num) / len(num)





