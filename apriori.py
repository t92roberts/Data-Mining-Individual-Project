import csv
import collections
import itertools

import data_mining_utilities


def load_attribute(full_data, attribute_name):
    """ Extract an attribute from the full dataset """
    attribute = []

    # iterate over each row, extract the value for the attribute
    for row in full_data:
        attribute.append(row[attribute_name])

    return attribute


def find_1_item_sets(transactions):
    """ Get the unique 1-itemsets """
    single_item_sets = []

    for transaction in transactions:
        for item in transaction:
            if [item] not in single_item_sets:  # only unique items
                single_item_sets.append([item])

    # convert to a frozenset to be hashable (to be used as a dictionary key)
    return list(map(frozenset, single_item_sets))



def scan(transactions, candidates):
    """ Find the support count of a set of candidates """
    support_counts = collections.defaultdict(lambda: 0)  # new candidates given a default support of 0

    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):  # candidate has support in the full list of transactions
                support_counts[candidate] += 1

    return support_counts



def discard(candidates, support_threshold):
    """ Discard candidates that don't have the minimum level of support """
    frequent_itemsets = []

    for itemset in candidates:
        if candidates[itemset] >= support_threshold:  # candidate has the minimum required support
            frequent_itemsets.append(list(itemset))

    return list(map(frozenset, frequent_itemsets))



def join(freq_itemsets, k):
    """ Generate a new set of k-item candidates """
    new_candidates = []

    num_of_sets = len(freq_itemsets)

    for i in range(num_of_sets):
        for j in range(i + 1, num_of_sets):  # don't compare the candidate with itself
            set_1 = list(freq_itemsets[i])
            set_1.sort()  # apriori depends on the item set being ordered
            start_set_1 = set_1[:-1]  # all elements except the last one

            set_2 = list(freq_itemsets[j])
            set_2.sort()
            start_set_2 = set_2[:-1]

            # combine the two itemsets if all elements except the last are the same
            if start_set_1 == start_set_2:
                new_candidate = freq_itemsets[i].union(freq_itemsets[j])  # combine the two (entire) item sets

                # prune candidates with infrequent subsets
                if not has_infrequent_subset(new_candidate, freq_itemsets, k):
                    new_candidates.append(new_candidate)

    return new_candidates


def has_infrequent_subset(new_candidate, freq_itemsets, k):
    """ Check if a generated candidate's subsets are all frequent """
    subsets = list(itertools.combinations(new_candidate, k - 1))  # generate subsets of the candidate of length k - 1

    for subset in subsets:
        if frozenset(subset) not in freq_itemsets:  # a subset is not frequent
            return True
        else:
            return False


def format_output(freq_items):
    """ Verbose output of the itemsets """
    # verbose output for testing purposes
    formatted = collections.OrderedDict()

    for k, items in enumerate(freq_items):
        list_items = list(map(list, items))

        for item in list_items:
            item.sort()

        list_items.sort()
        formatted["L" + str(k + 1)] = list_items

    return formatted


def apriori(transactions, min_abs_sup):
    """ Main apriori algorithm """
    frequent_itemsets = []  # list of all frequent k-itemsets
    frequent_itemsets_support = []  # support for  each frequent k-item set

    candidates_1 = find_1_item_sets(transactions)  # 1-itemsets
    print("k = 1 candidates:", len(candidates_1))
    support_1 = scan(transactions, candidates_1)  # 1-item support counts
    large_1 = discard(support_1, min_abs_sup)  # discard unsupported sets

    frequent_itemsets.append(large_1)  # first set of supported itemsets
    frequent_itemsets_support.append(scan(transactions, large_1))  # support for large_1

    print("k = 1 generated\n")

    k = 2

    # while the last itemsets have supporting subsets
    while len(frequent_itemsets[-1]) > 0:
        candidates_k = join(frequent_itemsets[k - 2], k)  # new candidates created from the last frequent item sets
        print("k =", k, "candidates:", len(candidates_k))
        support_k = scan(transactions, candidates_k)  # count the support
        large_k = discard(support_k, min_abs_sup)  # discard unsupported candidates

        frequent_itemsets.append(large_k)  # store k-th set of frequent itemsets
        frequent_itemsets_support.append(scan(transactions, large_k))  # support for large_k

        print("k =", k, "generated\n")

        k += 1

    return frequent_itemsets, frequent_itemsets_support


csv_file = csv.DictReader(open('Data Mining - Spring 2017.csv'))

# cleaned_data = [['A','C','D'], ['B','C'], ['A','B','C','E'], ['B','E']]  # example data from the slides

#data = load_attribute(csv_file, 'Which of these games have you played?')
#cleaned_data = pre_processing_2.clean_data_nominal_delimited(data, [";"])

data = load_attribute(csv_file, 'Which programming languages do you know?')
cleaned_data = data_mining_utilities.clean_data_nominal_delimited(data, [";", ",", " "])

min_sup = round(len(cleaned_data) * 0.2)  # relative minimum support
print("Minimum support =", min_sup, "\n")

freq_items, freq_items_support = apriori(cleaned_data, min_sup)
formatted = format_output(freq_items)

print("Results:\n")
for key, value in formatted.items():
    print(key, value, "\n")
