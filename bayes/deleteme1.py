!/usr/bin/env python
import os
import math
import glob
from collections import defaultdict, Counter

# Define paths to the datasets
MOVIE_REVIEW_TRAIN_PATH = 'movie_review/train/'
MOVIE_REVIEW_TEST_PATH = 'movie_review/test/'
NEWSGROUP_TRAIN_PATH = 'newsgroups/train/'
NEWSGROUP_TEST_PATH = 'newsgroups/test/'

def load_data(path, classes):
    data = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(path, cls)
        for filename in glob.glob(os.path.join(cls_path, '*.txt')):
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                data.append(text)
                labels.append(cls)
    return data, labels

def tokenize(text):
    # Simple whitespace tokenizer and lowercase
    tokens = text.lower().split()
    return tokens

def train_naive_bayes(data, labels):
    vocab = set()
    word_counts = {}
    class_counts = {}
    total_docs = len(labels)
    class_doc_counts = Counter(labels)

    # Initialize dictionaries
    for cls in set(labels):
        word_counts[cls] = defaultdict(int)
        class_counts[cls] = 0

    # Process each document
    for text, label in zip(data, labels):
        tokens = tokenize(text)
        class_counts[label] += len(tokens)
        for token in tokens:
            vocab.add(token)
            word_counts[label][token] += 1

    # Calculate priors
    priors = {}
    for cls in class_doc_counts:
        priors[cls] = math.log(class_doc_counts[cls] / total_docs)

    # Calculate likelihoods with Laplace smoothing
    likelihoods = {}
    V = len(vocab)
    for cls in word_counts:
        likelihoods[cls] = {}
        total_count = class_counts[cls]
        for word in vocab:
            count = word_counts[cls].get(word, 0)
            # Apply Laplace smoothing
            likelihoods[cls][word] = math.log((count + 1) / (total_count + V))

    return priors, likelihoods, vocab

def predict(text, priors, likelihoods, vocab):
    tokens = tokenize(text)
    classes = priors.keys()
    scores = {}
    for cls in classes:
        scores[cls] = priors[cls]
        for token in tokens:
            if token in vocab:
                scores[cls] += likelihoods[cls].get(token, math.log(1 / (sum([len(likelihoods[c]) for c in classes]) + len(vocab))))
            else:
                # Handle unknown words (Laplace smoothing)
                scores[cls] += math.log(1 / (sum([len(likelihoods[c]) for c in classes]) + len(vocab)))
    # Return the class with the highest score
    return max(scores, key=scores.get)

def evaluate(test_data, test_labels, priors, likelihoods, vocab):
    correct = 0
    total = len(test_labels)
    class_correct = Counter()
    class_total = Counter()
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for text, true_label in zip(test_data, test_labels):
        predicted_label = predict(text, priors, likelihoods, vocab)
        if predicted_label == true_label:
            correct += 1
            class_correct[true_label] += 1
        class_total[true_label] += 1
        confusion_matrix[true_label][predicted_label] += 1

    accuracy = correct / total

    # Calculate precision, recall for each class
    precision = {}
    recall = {}
    for cls in set(test_labels):
        tp = confusion_matrix[cls][cls]
        fp = sum([confusion_matrix[other][cls] for other in confusion_matrix if other != cls])
        fn = sum([confusion_matrix[cls][other] for other in confusion_matrix[cls] if other != cls])
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall, confusion_matrix

def print_evaluation_results(accuracy, precision, recall, confusion_matrix):
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Precision:")
    for cls in precision:
        print(f"  {cls}: {precision[cls]:.4f}")
    print("\nRecall:")
    for cls in recall:
        print(f"  {cls}: {recall[cls]:.4f}")
    print("\nConfusion Matrix:")
    classes = confusion_matrix.keys()
    header = "\t" + "\t".join(classes)
    print(header)
    for true_cls in classes:
        row = [true_cls]
        for pred_cls in classes:
            row.append(str(confusion_matrix[true_cls][pred_cls]))
        print("\t".join(row))

# Main execution for Movie Review dataset
movie_classes = ['pos', 'neg']
movie_train_data, movie_train_labels = load_data(MOVIE_REVIEW_TRAIN_PATH, movie_classes)
movie_test_data, movie_test_labels = load_data(MOVIE_REVIEW_TEST_PATH, movie_classes)

priors, likelihoods, vocab = train_naive_bayes(movie_train_data, movie_train_labels)
accuracy, precision, recall, confusion_matrix = evaluate(movie_test_data, movie_test_labels, priors, likelihoods, vocab)

print("Movie Review Dataset Evaluation:")
print_evaluation_results(accuracy, precision, recall, confusion_matrix)

# Main execution for Newsgroups dataset
newsgroup_classes = ['atheism', 'christian', 'misc']
newsgroup_train_data, newsgroup_train_labels = load_data(NEWSGROUP_TRAIN_PATH, newsgroup_classes)
newsgroup_test_data, newsgroup_test_labels = load_data(NEWSGROUP_TEST_PATH, newsgroup_classes)

priors, likelihoods, vocab = train_naive_bayes(newsgroup_train_data, newsgroup_train_labels)
accuracy, precision, recall, confusion_matrix = evaluate(newsgroup_test_data, newsgroup_test_labels, priors, likelihoods, vocab)

print("\nNewsgroups Dataset Evaluation:")
print_evaluation_results(accuracy, precision, recall, confusion_matrix)


