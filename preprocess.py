from typing import List, Dict
import heapq
from marketbasket.settings import settings
from marketbasket.transaction import Transaction
from collections import Counter
from datetime import datetime
from marketbasket.transactions_file import TransactionsFile
from marketbasket.feature import Feature
from marketbasket.labels import Labels

"""
    Preprocess data:
        * If some feature has a maximum number of labels, remove other labels
        * Remove transactions with a single item
        * Create label files
        * Print statistics about features
"""

print(datetime.now(), "Process start: Preprocess")
settings.print_summary()

# Number of label occurrences in transactions for each feature (key=Feature, value=counter with label n. of occurrences)
labels_occurrences:Dict[Feature, Counter] = {}
for feature in settings.features:
    # Counter (key = label, value = n. label occurrences)
    labels_occurrences[feature] = Counter()

# Count features label occurrences and total number of transactions
n_total_transactions = 0
n_transactions_multiple_items = 0
n_items_sells = 0
# Read transactions
with TransactionsFile(settings.transactions_file, 'r') as trn_file:
    for transaction in trn_file:
        n_total_transactions += 1

        # Ignore transactions with a single item
        if len(transaction.item_labels) < 2:
            continue

        n_transactions_multiple_items += 1

        # Count label occurrences for each transaction feature
        for feature in labels_occurrences:
            counter:Counter = labels_occurrences[feature]
            if feature.sequence:
                # Feature is sequence, count each label in sequence
                for label in transaction[feature.name]:
                    counter[label] += 1
            else:
                # Feature is unique for the entire transaction
                counter[ transaction[feature.name] ] += 1

        # Count number of item sells:
        n_items_sells += len(transaction.item_labels)

print("# original transactions:", n_total_transactions)
print("# transactions with n.items > 1:", n_transactions_multiple_items)
print("# original item sells (n.items > 1):", n_items_sells)
for feature in labels_occurrences:
    print("# original " + feature.name + " labels: " + str(len(labels_occurrences[feature])) )

def get_top_labels(occurrences: Counter, n_max: int) -> List[str]:
    """ Return labels most common in a Counter """
    return [pair[0] for pair in occurrences.most_common(n_max)]

# Now we have referenced labels. Store them as Labels instances in settings.features (FeaturesSet instance)
for feature in labels_occurrences:
    feature_labels_counter:Counter = labels_occurrences[feature]

    original_labels = feature_labels_counter.keys()
    if feature.max_labels > 0:
        # Feature has limited it's number of labels. Get top labels:
        label_values = get_top_labels(feature_labels_counter, feature.max_labels)
    else:
        label_values = original_labels

    feature.labels = Labels(label_values)

    if len(label_values) < len(original_labels) and feature.name != settings.features.item_label_feature:
        # There will be unknown labels. They will be replace by a "[UNKNOWN]" label
        # For item labels there is an exception: They will be ignored
        feature.labels.append(Labels.UNKNOWN_LABEL)

# Filter transactions: Remove non top labels, remove transactions with a single item
n_final_transactions = 0
n_final_item_sells = 0
sequences_lengths = Counter()
with TransactionsFile(settings.transactions_file, 'r') as trn_file:
    with TransactionsFile(Transaction.top_items_path(), 'w') as trn_top_file:
        for transaction in trn_file:

            # Remove features out of top labels
            for feature in settings.features:
                if feature.max_labels > 0:
                    transaction[feature.name] = feature.filter_wrong_labels(transaction[feature.name])

            # Ignore empty and single item transactions 
            n_items = len(transaction.item_labels)
            if n_items > 1:
                n_final_transactions += 1
                n_final_item_sells += n_items
                sequences_lengths[n_items] += 1

                trn_top_file.write(transaction)

settings.features.save_label_files()

print()
print("# final transactions:", n_final_transactions )
print("# item sells (final):", n_final_item_sells )
if n_items_sells > 0:
    print("Ratio item sells supported:", str(n_final_item_sells/n_items_sells) )

# print("Sequences lengths:")
# for seq_length in sorted(sequences_lengths.keys()):
#     print(seq_length, sequences_lengths[seq_length])

sum_of_numbers = sum(number*count for number, count in sequences_lengths.items())
count = sum(count for n, count in sequences_lengths.items())
mean = sum_of_numbers / count
print("Mean sequence length:", mean)

print(datetime.now(), "Process end")
