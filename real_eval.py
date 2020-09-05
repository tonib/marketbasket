from labels import Labels
from settings import Settings
from predict import Prediction
from transaction import Transaction
from typing import List, Tuple, Iterable
import cProfile
import re

TEST_BATCH_SIZE = 256

prediction = Prediction()

def transactions_with_expected_item() -> Iterable[Tuple[Transaction, str]]:
    print("transactions_with_expected_item started")
    with open(Transaction.TRANSACTIONS_EVAL_DATASET_FILE) as eval_trn_file:
        for line in eval_trn_file:
            transaction = Transaction(line)

            for idx in range(1, len(transaction.item_labels)):
                input_items_labels = transaction.item_labels[0:idx]
                expected_item_label = transaction.item_labels[idx]
                input_transaction = Transaction.from_labels(input_items_labels, transaction.customer_label)
                #print("yield trn")
                yield ( input_transaction , expected_item_label )

def transactions_with_expected_item_batches() -> Iterable[ List[Tuple[Transaction, str]] ]:
    print("transactions_with_expected_item_batches started")
    batch = []
    for trn in transactions_with_expected_item():
        #print("got trn")
        batch.append( trn )
        if len( batch ) >= TEST_BATCH_SIZE:
            #print("yield batch")
            yield batch
            batch = []
    # Last batch
    if len(batch) > 0:
        yield batch

def run_eval():
    score = 0
    n_predictions = 0
    for batch in transactions_with_expected_item_batches():

        #print("got batch")

        input_batch = [ trn_with_expected[0] for trn_with_expected in batch ]

        results = prediction.predict_batch(input_batch, 10)

        for idx, transaction_with_expected_result in enumerate(batch):

            # Get predicted items
            #print(results)
            predicted_item_labels = [ p[0] for p in results[idx] ]

            n_predictions += 1
            if n_predictions % 1000 == 0:
                print(n_predictions)
            
            expected_item = transaction_with_expected_result[1]
            if expected_item in predicted_item_labels:
                score += +1

    print("Score: " + str(score) + " of " + str(n_predictions))

#run_eval()
cProfile.run('run_eval()')

