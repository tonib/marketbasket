from typing import List, Dict
from operator import itemgetter
import heapq
from labels import Labels
from settings import settings
from transaction import Transaction
from collections import Counter
from datetime import datetime

print(datetime.now(), "Process start: Preprocess")
settings.print_summary()

# Number of item/customer ocurrences in transactions (key = item key, value = n. ocurrences)
items_occurrences = Counter()
customers_occurrences = Counter()

def get_top_labels(occurrences: Counter, n_max: int) -> List[str]:
    return [pair[0] for pair in occurrences.most_common(n_max)]

n_transactions = 0
n_total_item_sells = 0
with open(settings.transactions_file) as trn_file:
    for line in trn_file:
        transaction = Transaction(line)

        # Remove duplicated items
        transaction.remove_duplicated_items()

        # Ignore single item transactions (not useful to search relations...)
        if len(transaction.item_labels) > 1:
            n_transactions += 1
            for item in transaction.item_labels:
                items_occurrences[item] += 1
                n_total_item_sells += 1
            customers_occurrences[transaction.customer_label] += 1

print("# transactions with more than one item:", n_transactions )
print("# item sells (trn. with more than one item):", n_total_item_sells )
print("# total items:", len(items_occurrences))
print("# total customers:", len(customers_occurrences))

# Save top item/customer labels
item_labels = Labels( get_top_labels(items_occurrences, settings.n_max_items) )
customer_labels = Labels( get_top_labels(customers_occurrences, settings.n_max_customers) )

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

            # Remove duplicated items
            transaction.remove_duplicated_items()

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
