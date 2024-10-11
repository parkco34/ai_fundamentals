#!/usr/bin/env python
import csv
import string
from collections import defaultdict
import math

# Step 1: Load the data
def load_data(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append((row[0], row[1]))  # Assuming row[0] is the text, row[1] is the label (pos/neg)
    return data

# Step 2: Preprocess text (tokenize and remove punctuation)
def preprocess(text):
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower().split()

# Step 3: Build vocabulary and calculate word counts for each class
def build_vocab_and_counts(data):
    vocab = defaultdict(int)
    class_word_counts = {'pos': defaultdict(int), 'neg': defaultdict(int)}
    class_totals = {'pos': 0, 'neg': 0}

    for text, label in data:
        words = preprocess(text)
        for word in words:
            vocab[word] += 1
            class_word_counts[label][word] += 1
            class_totals[label] += 1

    return vocab, class_word_counts, class_totals

# Step 4: Calculate the log-probability of a class with Laplace smoothing
def calculate_class_probabilities(words, class_word_counts, class_totals, vocab_size, class_label):
    total_words_in_class = class_totals[class_label]
    log_prob = 0.0

    for word in words:
        word_count = class_word_counts[class_label].get(word, 0)
        # Apply Laplace smoothing
        prob_word_given_class = (word_count + 1) / (total_words_in_class + vocab_size)
        log_prob += math.log(prob_word_given_class)

    return log_prob

# Step 5: Classify a new text
def classify(text, class_word_counts, class_totals, vocab_size):
    words = preprocess(text)

    pos_prob = calculate_class_probabilities(words, class_word_counts, class_totals, vocab_size, 'pos')
    neg_prob = calculate_class_probabilities(words, class_word_counts, class_totals, vocab_size, 'neg')

    # Compare probabilities
    if pos_prob > neg_prob:
        return 'pos'
    else:
        return 'neg'

# Step 6: Evaluate the model
def evaluate(data, class_word_counts, class_totals, vocab_size):
    correct = 0
    total = len(data)

    for text, actual_label in data:
        predicted_label = classify(text, class_word_counts, class_totals, vocab_size)
        if predicted_label == actual_label:
            correct += 1

    return correct / total

# Main function
def main():
    # Load the training data
    training_data = load_data('./data/reviews_polarity_train.csv')  # Update this to your dataset path

    # Build vocabulary and word counts
    vocab, class_word_counts, class_totals = build_vocab_and_counts(training_data)
    vocab_size = len(vocab)

    # Load the test data
    test_data = load_data('./data/reviews_polarity_test.csv')  # Update this to your test dataset path

    # Evaluate the model
    accuracy = evaluate(test_data, class_word_counts, class_totals, vocab_size)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()


