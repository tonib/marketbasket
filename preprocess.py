from typing import List, Dict
from operator import itemgetter
import heapq
from product_labels import ProductLabels

# Max number of items to handle
N_MAX_ITEMS = 100

# Number of item ocurrences in transactions (key = item key, value = n. ocurrences)
items_instances: Dict[str, int] = {}

n_transactions = 0
with open('data/transactions.txt') as trn_file:
    for line in trn_file:
        transaction_items: List[str] = line.split()
        # Ignore single item transactions (not useful to search relations...)
        if len(transaction_items) > 1:
            n_transactions += 1
            for item in transaction_items:
                if item in items_instances:
                    items_instances[item] += 1
                else:
                    items_instances[item] = 1

print("# transactions with more than one item:", n_transactions )
print("# items:", len(items_instances))

# Get items with max number of ocurrences (idx 0 = item description, idx = 1, nº instances)
top_item_ocurrences: Dict[str, int] = dict( heapq.nlargest(N_MAX_ITEMS, items_instances.items(), key=itemgetter(1)) )

# Save top item codes
product_labels = ProductLabels( top_item_ocurrences.keys() )
product_labels.save()

# Filter transactions
with open('data/transactions.txt') as trn_file:
    with open('data/transactions_top_items.txt', 'w') as trn_top_file:

        for line in trn_file:
            transaction_items: List[str] = line.split()

            # Keep top items only, replacing item codes by its index
            top_items_transaction: List[str] = []
            for item in transaction_items:
                if item in product_labels.indices:
                    top_items_transaction.append( str(product_labels.indices[item]) )
            
            if len(top_items_transaction) > 1:
                # Ignore single item transactions (not useful to search relations...)
                trn_top_file.write( ' '.join(top_items_transaction) + '\n' )

