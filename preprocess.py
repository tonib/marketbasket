from typing import List, Dict
from operator import itemgetter
import heapq
from labels import Labels
from settings import Settings
from transaction import Transaction

# Number of item/customer ocurrences in transactions (key = item key, value = n. ocurrences)
items_occurrences: Dict[str, int] = {}
customers_occurrences: Dict[str, int] = {}

def get_top_labels(occurrences: Dict[str, int], n_max: int) -> List[str]:
    top_ocurrences: Dict[str, int] = dict( heapq.nlargest(n_max, occurrences.items(), key=itemgetter(1)) )
    return top_ocurrences.keys()
    
n_transactions = 0
n_total_item_sells = 0
with open('data/transactions.txt') as trn_file:
    for line in trn_file:
        transaction = Transaction(line)

        # Remove duplicated items
        transaction.remove_duplicated_items()

        # Ignore single item transactions (not useful to search relations...)
        if len(transaction.item_labels) > 1:
            n_transactions += 1
            for item in transaction.item_labels:
                if item in items_occurrences:
                    items_occurrences[item] += 1
                else:
                    items_occurrences[item] = 1
                n_total_item_sells += 1

            if transaction.customer_label in customers_occurrences:
                customers_occurrences[transaction.customer_label] += 1
            else:
                customers_occurrences[transaction.customer_label] = 1

print("# transactions with more than one item:", n_transactions )
print("# item sells (trn. with more than one item):", n_total_item_sells )
print("# total items:", len(items_occurrences))
print("# total customers:", len(customers_occurrences))

# Save top item/customer labels
item_labels = Labels( get_top_labels(items_occurrences, Settings.N_MAX_ITEMS) )
customer_labels = Labels( get_top_labels(customers_occurrences, Settings.N_MAX_CUSTOMERS) )

# Filter transactions
n_transactions = 0
n_transactions_with_customer_id = 0
n_final_item_sells = 0
there_are_unknown_customers = False
with open('data/transactions.txt') as trn_file:
    with open(Transaction.TRANSACTIONS_TOP_ITEMS_PATH, 'w') as trn_top_file:
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


item_labels.save(Labels.ITEM_LABELS_FILE)
if there_are_unknown_customers and not customer_labels.contains(Labels.UNKNOWN_LABEL):
    customer_labels.append(Labels.UNKNOWN_LABEL)
customer_labels.save(Labels.CUSTOMER_LABELS_FILE)

print()
print("# top items:", len(item_labels.labels) )
print("# top customers:", len(customer_labels.labels) )
print("# final transactions:", n_transactions )
print("# transactions with customer id:", n_transactions_with_customer_id )
print("# item sells (final):", n_final_item_sells )
if n_total_item_sells > 0:
    print("Ratio item sells supported:", str(n_final_item_sells/n_total_item_sells) )
