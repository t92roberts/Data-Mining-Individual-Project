import collections
import re
import math


def histogram_nominal(values):
    frequencies = collections.defaultdict(lambda: 0)
    for value in values:
        frequencies[value] += 1
    return frequencies


def min_max_normalise(numbers, new_min, new_max):
    min_number = min(numbers)
    max_number = max(numbers)

    normalised = numbers.copy()

    for index, number in enumerate(normalised):
        normalised[index] = ((number - min_number) / (max_number - min_number)) * (new_max - new_min) + new_min

    return normalised


def clean_data_nominal_delimited(data, delimiters_list):
    # construct the regex from the list of delimiters
    delimiters = ""
    for delimiter in delimiters_list:
        delimiters += delimiter + "|"

    delimiters = delimiters[:-1]  # remove the trailing pipe
    p = re.compile(delimiters)

    cleaned_attribute = []
    for row in data:
        split = p.split(row)

        cleaned_row = []
        for item in split:
            if item != "":  # ignore empty strings created by the splitting
                item = item.strip()  # strip leading/trailing whitespace

                # convert alphabetical characters to lower case
                cleaned_string = ""
                for char in item:
                    if char.isalpha():
                        char = char.lower()
                    cleaned_string += char

                cleaned_row.append(cleaned_string)

        cleaned_attribute.append(cleaned_row)

    return cleaned_attribute


def mean(numbers):
    return sum(numbers) / len(numbers)


def trimmed_mean(numbers, percentage_trimmed):
    sorted_numbers = numbers.copy()
    sorted_numbers.sort()

    # convert relative trim to absolute trim
    num_to_trim = math.floor(len(sorted_numbers) * percentage_trimmed)

    # discard values from the start and end of the set of values
    sorted_numbers = sorted_numbers[num_to_trim: - 1 * num_to_trim]

    if len(sorted_numbers) > 0:
        return mean(sorted_numbers)
    else:
        # all values were trimmed, return the mean of the untrimmed set of values
        return mean(numbers)


def variance(vectors, mean):
    squared_differences = 0
    for vector in vectors:
        difference = euclidean_distance(vector, mean)
        try:
            squared_differences += pow(difference, 2)
        except TypeError as e:
            print(e)
    return squared_differences / float(len(vectors))


def euclidean_distance(vector_1, vector_2):
    if len(vector_1) != len(vector_2):
        return "Error - vectors are different lengths"

    distance = 0
    for attribute in zip(vector_1, vector_2):
        distance += pow(attribute[0] - attribute[1], 2)

    return math.sqrt(distance)
