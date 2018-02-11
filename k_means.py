import csv
import operator
import random
import collections

import data_mining_utilities


def load_attributes(full_data):
    """ Extract and clean the age, shoe size, and height attributes """
    ages = []
    shoe_sizes = []
    heights = []

    for row in full_data:
        subset = []

        # exception handling to catch any non-numeric data before cleaning it
        try:
            subset.append(float(row['Age'].replace(",", ".")))
        except ValueError as e:
            print(e)

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
            ages.append(subset[0])
            shoe_sizes.append(subset[1])
            heights.append(subset[2])

    return ages, shoe_sizes, heights


def initial_centroids(vectors, k):
    """ Randomly select k data points to be the initial set of centroids """
    random_vectors = random.sample(vectors, k)  # random sampling of data points (without replacement)

    clusters = collections.OrderedDict()
    for i, vector in enumerate(random_vectors):  # assign each centroid an arbitrary ID number
        clusters[i] = vector

    return clusters


def initial_clusters(vectors, centroids):
    """ Assign all of the unlabelled data points to their nearest centroid """
    initial_clusters = []

    # calculate the Euclidean distance from each data point to every centroid
    for vector in vectors:
        distances = {}
        for cluster_id, centroid in centroids.items():
            distances[cluster_id] = data_mining_utilities.euclidean_distance(vector, centroid)

        # assign the cluster ID of the nearest centroid to the data point
        nearest_centroid = min(distances.items(), key=operator.itemgetter(1))[0]
        initial_clusters.append([nearest_centroid, vector])

    return initial_clusters


def get_vectors_in_clusters(labelled_vectors):
    """ Group the data points by the cluster they're assigned to """
    clusters = collections.defaultdict(lambda: list())
    for vector in labelled_vectors:
        clusters[vector[0]] += [vector[1]]
    return clusters


def update_centroids(labelled_vectors):
    """ Update the cluster centroids to be the mean of the data points in the cluster """
    new_centroids = {}

    clusters = get_vectors_in_clusters(labelled_vectors)  # every data point grouped by cluster ID

    for cluster_id, vectors_in_cluster in clusters.items():
        attribute_values = []
        for i in range(len(labelled_vectors[0][1])):
            attribute_values.append([vector[i] for vector in vectors_in_cluster])

        # update the centroid of a cluster to be the trimmed mean of the data points in the cluster
        attribute_means = []
        for attribute in attribute_values:
            attribute_means.append(data_mining_utilities.trimmed_mean(attribute, 0.1))

            new_centroids[cluster_id] = attribute_means

    return new_centroids


def update_vectors(centroids, vectors):
    """ Update each data point's cluster label to the centroid nearest to it """
    updated_vector_labels = []

    # calculate the Euclidean distance from each data point to every centroid
    for vector in vectors:
        distances = {}
        for cluster_id, centroid in centroids.items():
            distances[cluster_id] = data_mining_utilities.euclidean_distance(vector[1], centroid)

        # update the cluster ID of the data point to the nearest centroid
        nearest_centroid = min(distances.items(), key=operator.itemgetter(1))[0]
        updated_vector_labels.append([nearest_centroid, vector[1]])

    return updated_vector_labels


def cluster_quality(centroids, labelled_vectors):
    """ Calculate the average intra-cluster variance of the clusters """
    clusters = get_vectors_in_clusters(labelled_vectors)

    intra_cluster_variance = []
    for cluster_id, centroid in centroids.items():
        cluster_quality = data_mining_utilities.variance(clusters[cluster_id], centroid)
        intra_cluster_variance.append(cluster_quality)

    return data_mining_utilities.mean(intra_cluster_variance)


def k_means(initial_centroids, initial_labelled_vectors, k):
    """ Main k-means algorithm """
    # initialise the first centroids and unlabelled data points
    centroids = initial_centroids
    labelled_vectors = initial_labelled_vectors

    average_centroid_movement = float("inf")  # initialise to infinity

    # repeat until the algorithm reaches a local minimum
    while average_centroid_movement > 0:
        # 'move' the centroids to better represent their cluster's centre
        new_centroids = update_centroids(labelled_vectors)

        differences = []
        for i in range(k):
            if i in new_centroids.keys():  # a cluster may have no vectors in it
                # calculate how far each centroid moved since the last iteration
                differences.append(data_mining_utilities.euclidean_distance(centroids[i], new_centroids[i]))

        # average centroid movement
        average_centroid_movement = data_mining_utilities.mean(differences)

        # update the centroids and labels of data points
        centroids = new_centroids
        labelled_vectors = update_vectors(new_centroids, labelled_vectors)

    return centroids, labelled_vectors


file = csv.DictReader(open('Data Mining - Spring 2017.csv'))
ages, shoe_sizes, heights = load_attributes(file)  # apply cleaning rules

""" Normalise numeric data using min/max """
new_min = -1
new_max = 1
ages = data_mining_utilities.min_max_normalise(ages, new_min, new_max)
shoe_sizes = data_mining_utilities.min_max_normalise(shoe_sizes, new_min, new_max)
heights = data_mining_utilities.min_max_normalise(heights, new_min, new_max)

""" Recombine the cleaned/normalised attributes into 3-dimensional vectors """
normalised_vectors = []
for vector in zip(ages, shoe_sizes, heights):
    normalised_vectors.append(vector)

""" Run k-means using incrementing values of k 
    (each iteration uses the same initial centroids created above, for consistency) """
min_k = 1
max_k = len(normalised_vectors)
starting_centroids = initial_centroids(normalised_vectors, max_k)

for k in range(min_k, max_k + 1):
    """ Set the first centroids to be the random initial centroids """
    k_centroids = collections.OrderedDict()
    for i in range(k):
        k_centroids[i] = starting_centroids[i]

    """ Label the data points with their closest centroid's label """
    initial_labelled_vectors = initial_clusters(normalised_vectors, k_centroids)

    """ Run k-means with the initial clusters and labelled vectors """
    centroids, labelled_vectors = k_means(k_centroids, initial_labelled_vectors, k)

    """ Evaluate the quality of the clusters by calculating the variance within the clusters """
    average_intra_cluster_variance = cluster_quality(centroids, labelled_vectors)
    print(k, average_intra_cluster_variance)
