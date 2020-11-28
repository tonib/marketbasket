from typing import List, Dict
from operator import itemgetter
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
n_transactions = 0
n_items_sells = 0
# Read transactions
with TransactionsFile(settings.transactions_file, 'r') as trn_file:
    for transaction in trn_file:
        n_transactions += 1

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

print("# original transactions:", n_transactions)
print("# original item sells:", n_items_sells)
for feature in labels_occurrences:
    print("# original " + feature.name + " labels: " + str(len(labels_occurrences[feature])) )

def get_top_labels(occurrences: Counter, n_max: int) -> List[str]:
    """ Return labels most common in a Counter """
    return [pair[0] for pair in occurrences.most_common(n_max)]

# Now we have referenced labels. Store them as Labels instances in settings.features (FeaturesSet instance)
for feature in labels_occurrences:
    feature_labels_counter:Counter = labels_occurrences[feature]

    if feature.max_labels > 0:
        # Feature has limited it's number of labels. Get top labels:
        label_values = get_top_labels(feature_labels_counter, feature.max_labels)
    else:
        label_values = feature_labels_counter.keys()

    feature.labels = Labels( label_values )
exit()


# Filter transactions
n_transactions = 0
n_transactions_with_customer_id = 0
n_final_item_sells = 0
there_are_unknown_customers = False
sequences_lengths = Counter()
with open(settings.transactions_file) as trn_file:
    with open(Transaction.top_items_path(), 'w') as trn_top_file:
        for line in trn_file:
            transaction = Transaction(line)

            # Keep top items only
            top_items_transaction: List[str] = []
            for item in transaction.item_labels:
                if item_labels.contains(item):
                    top_items_transaction.append( item )
            transaction.item_labels = top_items_transaction

            # Ignore single item transactions (not useful to search relations...)
            n_items_trn = len(transaction.item_labels)
            if n_items_trn > 1:
                n_final_item_sells += n_items_trn
                #transaction.assert_no_duplicates()

                # Keep top customer only
                if not customer_labels.contains(transaction.customer_label):
                    transaction.customer_label = Labels.UNKNOWN_LABEL
                    there_are_unknown_customers = True
                else:
                    n_transactions_with_customer_id += 1

                trn_top_file.write( str(transaction) + '\n' )
                n_transactions += 1
                sequences_lengths[n_items_trn] += 1

item_labels.save(Labels.item_labels_path())
if there_are_unknown_customers and not customer_labels.contains(Labels.UNKNOWN_LABEL):
    customer_labels.append(Labels.UNKNOWN_LABEL)
customer_labels.save(Labels.customer_labels_path())

print()
print("# top items:", len(item_labels.labels) )
print("# top customers:", len(customer_labels.labels) )
print("# final transactions:", n_transactions )
print("# transactions with customer id:", n_transactions_with_customer_id )
print("# item sells (final):", n_final_item_sells )
if n_total_item_sells > 0:
    print("Ratio item sells supported:", str(n_final_item_sells/n_total_item_sells) )

# print("Sequences lengths:")
# for seq_length in sorted(sequences_lengths.keys()):
#     print(seq_length, sequences_lengths[seq_length])

sum_of_numbers = sum(number*count for number, count in sequences_lengths.items())
count = sum(count for n, count in sequences_lengths.items())
mean = sum_of_numbers / count
print("Mean sequence length:", mean)

print(datetime.now(), "Process end")
