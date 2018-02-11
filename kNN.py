import csv
import math
from operator import itemgetter
import collections

import data_mining_utilities


def ad_hoc_analysis():
    """ Histogram of genders (used to create cleaning rules in the 'clean_gender' function) """
    raw_genders = [row['Gender'] for row in csv.DictReader(open('Data Mining - Spring 2017.csv'))]
    gender_frequencies = data_mining_utilities.histogram_nominal(raw_genders)
    return gender_frequencies


def clean_gender(raw_gender):
    """ Discrepancy correction using ad-hoc cleaning rules """
    if raw_gender in ['female', 'f']:
        return 'female'
    elif raw_gender in ['male', 'man', 'm']:
        return 'male'
    else:
        return None


def load_attributes(full_data):
    """ Load the gender, shoe size and height data """
    genders = []
    shoe_sizes = []
    heights = []

    for row in full_data:
        subset = []

        gender = clean_gender(row['Gender'].lower())
        if gender is not None:  # if gender is one of the accepted values
            subset.append(gender)

        # exception handling to catch any non-numeric data before cleaning it
        try:
            subset.append(float(row['Shoe Size'].replace(",", ".")))
        except ValueError as e:
            print(e)

        try:
            subset.append(float(row['Height'].replace(",", ".")))
        except ValueError as e:
            print(e)

        # keep the vector if all three attributes were successfully cleaned
        # (discard the vector if not)
        if len(subset) == 3:
            genders.append(subset[0])
            shoe_sizes.append(subset[1])
            heights.append(subset[2])

    return genders, shoe_sizes, heights


def get_neighbours(training_data, test_vector, k):
    """ Get the k nearest neighbours to a test data point """
    distances = []

    for vector in training_data:
        distance_to_new_data = data_mining_utilities.euclidean_distance(test_vector, vector[1:])
        distances.append((vector, distance_to_new_data))

    distances = sorted(distances, key=itemgetter(1))  # sort by distance to the vector being tested

    nearest_neighbours = []
    for i in range(k):  # return the k nearest neighbours
        nearest_neighbours.append(distances[i])

    return nearest_neighbours


def vote(nearest_neighbours):
    """ Get the most frequent label in a set of nearest neighbours """
    votes = collections.defaultdict(lambda: 0)

    for neighbour in nearest_neighbours:
        votes[neighbour[0][0]] += 1

    votes = sorted(votes.items(), key=itemgetter(1))  # sort by number of votes each label had
    return votes[-1][0]  # return label with the most votes


def show_accuracy(training_data, test_data, min_k, max_k):
    """ Verbose output of the prediction accuracy for each value of k """
    for k in range(min_k, max_k + 1):
        print("k =", str(k) + ":", str(accuracy(training_data, test_data, k)) + "% accuracy")

def accuracy(training_data, test_data, k):
    """ Invoke k-NN algorithm using the test data """
    correct_predictions = 0

    for test_vector in test_data:
        predicted_label = kNN(training_data, test_vector[1:], k)  # invoke the k-NN classification
        if predicted_label == test_vector[0]:  # correctly predicted the test data
            correct_predictions += 1

    return (correct_predictions / len(test_data)) * 100.0  # percentage of test vectors correctly predicted


def kNN(training_data, test_vector, k):
    """ Main k-nearest neighbour function """
    nearest_neighbours = get_neighbours(training_data, test_vector, k)
    predicted_label = vote(nearest_neighbours)
    return predicted_label

#print(ad_hoc_analysis())  # initial ad-hoc analysis of nominal data

file = csv.DictReader(open('Data Mining - Spring 2017.csv'))
genders, shoe_sizes, heights = load_attributes(file)  # apply cleaning rules

# normalise numeric data
new_min = -1
new_max = 1
shoe_sizes = data_mining_utilities.min_max_normalise(shoe_sizes, new_min, new_max)
heights = data_mining_utilities.min_max_normalise(heights, new_min, new_max)

# recombine the cleaned/normalised attributes into 3-dimensional vectors
normalised_vectors = []
for vector in zip(genders, shoe_sizes, heights):
    normalised_vectors.append(vector)

# split data into 2/3 training data, 1/3 testing data
split_point = math.floor(len(normalised_vectors) * 2 / 3)
training_data = normalised_vectors[:split_point]
test_data = normalised_vectors[split_point:]

# test the prediction accuracy for different values of k
minimum_k = 1
maximum_k = len(training_data)
show_accuracy(training_data, test_data, minimum_k, maximum_k)
