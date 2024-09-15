#!/usr/bin/env python
"""
Help from: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""
from math import exp, sqrt, pi

def read_data(filename):
    """
    Reading csv file only to save meself some time ( ° ͜ʖ °)
    -----------------------------------
    INPUT:
        filename: (str)

    OUTPUT:
        dataset: (list)
    """
    with open(filename, "r") as file:
        csv = file.readlines()

    # Parse data
    columns = csv[0].strip().split(",")

    dataset = []
    # Cuz we already got the column (first line)
    for line in csv[1:]:
        values = line.strip().split(",")
        features = [float(x) for x in values[:-1]]
        label = values[-1].strip()
        dataset.append(features + [label])
    
    return dataset

def split_data(dataset):
    """
    splits data into training and test sets.
    ------------------------------------------------
    input:
        dataset: (list)
   
    output:
        features, classes: (tuple of lists)
    """
    features, classes = [], []

    for inst in dataset:
        # Convert features to float
        features.append([float(x) for x in inst[:-1]])
        classes.append(inst[-1])

    return features, classes

# Mean3
def mean(num):
    return sum(num) / len(num)

# Variance
def variance(nums, mean_value):
    samples = len(nums)
    if samples < 2:
        return 0.0

    return sum((x - mean_value)**2 for x in nums) / (samples-1)

def summarize_data_classes(features, classes):
    """
    Summarizes dataset via calculating mean/variance per feature/class.
    ------------------------------------------------
    INPUT:
        features: (list of lists)
        classes: (list)
    
    OUTPUT:
        summaries: (dict)
    """
    summaries = {}

    # Combine features and classes
    for i in range(len(classes)):
        class_value = classes[i]
        vector = features[i]

        if class_value not in summaries:
            summaries[class_value] = [[] for _ in range(len(vector))]

        for j in range(len(vector)):
            summaries[class_value][j].append(vector[j])

    # Statistics
    for class_value, feature_lists in summaries.items():
        
        for i in range(len(feature_lists)):
            mew = mean(feature_lists[i])
            var = variance(feature_lists[i], mew)
            feature_lists[i] = (mew, var)

    return summaries

# Gaussian Prob Desnity
def probability(x, mean_value, variance_value):
    """
    Somthing to explain some shit... lolz.
    """
    if variance_value == 0:
        return 1.0 if x == mean_value else 0.0

    exponent = exp(-((x - mean_value)**2) / (2 * variance_value))

    return (1 / sqrt(2 * pi * variance_value)) * exponent

def class_probabilities(summaries, input_vector, priors):
    """
    Some shit...
    """
    probs = {}
    for class_value, class_summaries in summaries.items():
        probs[class_value] = priors[class_value]

        for i in range(len(class_summaries)):
            mean_value, variance_value = class_summaries[i]
            x = input_vector[i]
            prob = probability(x, mean_value, variance_value)

            probs[class_value] *= prob

    return probs

def prediction(summaries, input_vector, priors):
    """
    Hi
    """
    probs = class_probabilities(summaries, input_vector, priors)
    best_label, best_prob = None, -1

    for class_value, prob in probs.items():
        if best_label is None or prob > best_prob:
            best_prob = prob
            best_label = class_value

    return best_label

if __name__ == "__main__":
    dataset = read_data("iris_dataset.csv")

    features, labels = split_data(dataset)
    summaries = summarize_data_classes(features, labels)
    
    # Compute priors
    priors = {}
    total_instances = len(features)
    class_counts = {}

    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    for class_value in class_counts:
        priors[class_value] = class_counts[class_value] / float(total_instances)

    # Evalutation time!
    correct = 0
    for i in range(len(features)):
        input_vector = features[i]
        actual_label = labels[i]
        predicted_label = prediction(summaries, input_vector, priors)

        if predicted_label == actual_label:
            correct += 1

    accuracy = correct / len(features) * 100.0
    print(f"Accuracy on training data: {accuracy:.2f}")
    # Predict from new input file
    results = predict_from_file("/mnt/data/new_input.csv", summaries, priors)

    # Output the results
    for input_vector, predicted_class in results:
        print(f"Input: {input_vector} -> Predicted class: {predicted_class}")

