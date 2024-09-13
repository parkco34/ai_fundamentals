#!/usr/bin/env python
import math

# Read data from 'iris_dataset.csv'
with open('iris_dataset.csv', 'r') as file:
    lines = file.readlines()

# Parse the data
header = lines[0].strip().split(',')

dataset = []
for line in lines[1:]:
    values = line.strip().split(',')
    # turns string into float, 
    features = [float(x) for x in values[:-1]]
    # Target value
    label = values[-1]
    # Concatenate the target value to feature
    dataset.append(features + [label])

# Separate data by class
classes = {}
for instance in dataset:
    # Iterates thru each line of the data
    label = instance[-1]
    if label not in classes:
        classes[label] = [] # ?
    classes[label].append(instance[:-1])  # Exclude the label

# Function to calculate the mean
def mean(numbers):
    return sum(numbers) / len(numbers)

# Function to calculate the variance
def variance(numbers, mean_value):
    return sum((x - mean_value) ** 2 for x in numbers) / (len(numbers) - 1)

# Summarize the dataset by class
def summarize_dataset_by_class(classes):
    summaries = {}
    for class_value, instances in classes.items():
        features = list(zip(*instances))
        summaries[class_value] = []
        for i in range(len(features)):
            feature = features[i]
            m = mean(feature)
            v = variance(feature, m)
            summaries[class_value].append((m, v))
    return summaries

summaries = summarize_dataset_by_class(classes)

# Compute prior probabilities
priors = {}
total_instances = len(dataset)
for class_value in classes:
    priors[class_value] = len(classes[class_value]) / float(total_instances)

# Gaussian Probability Density Function
def calculate_probability(x, mean_value, variance_value):
    exponent = math.exp(- ((x - mean_value) ** 2) / (2 * variance_value))
    return (1 / math.sqrt(2 * math.pi * variance_value)) * exponent

# Calculate class probabilities for an input vector
def calculate_class_probabilities(summaries, input_vector, priors):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = priors[class_value]
        for i in range(len(class_summaries)):
            mean_value, variance_value = class_summaries[i]
            x = input_vector[i]
            # Handle the case where variance is zero
            if variance_value == 0:
                if x == mean_value:
                    probability = 1.0
                else:
                    probability = 0.0
            else:
                probability = calculate_probability(x, mean_value, variance_value)
            probabilities[class_value] *= probability
    return probabilities

# Make a prediction for a given input vector
def predict(summaries, input_vector, priors):
    probabilities = calculate_class_probabilities(summaries, input_vector, priors)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# Evaluate the model on the training data
correct = 0
for instance in dataset:
    input_vector = instance[:-1]
    actual_label = instance[-1]
    predicted_label = predict(summaries, input_vector, priors)
    if predicted_label == actual_label:
        correct += 1

accuracy = correct / len(dataset) * 100.0
print('Accuracy on training data: %.2f%%' % accuracy)

# Example of making a prediction on new data
new_data = [6.1, 3.0, 4.6, 1.4]
predicted_class = predict(summaries, new_data, priors)
print('Predicted class for input {}: {}'.format(new_data, predicted_class))


