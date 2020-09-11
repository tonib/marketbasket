from labels import Labels
from settings import Settings
from predict import Prediction
from transaction import Transaction
from typing import List, Tuple, Iterable
import cProfile
import re
from pstats import SortKey
import time
import tensorflow as tf

TEST_BATCH_SIZE = 256

# Test
#tf.config.run_functions_eagerly(True)
# predictor = Prediction()
# r = predictor.predict_single( Transaction.from_labels( ['4333' ], '[UNKNOWN]' ) , 10 )
# print(r)
# exit()

# Test
#tf.config.run_functions_eagerly(True)
# predictor = Prediction()
# batch = [ Transaction.from_labels( ['4333', '21730' ], '[UNKNOWN]' ) , Transaction.from_labels( ['4333' ], '[UNKNOWN]' ) ]
# r = predictor.predict_batch( batch , 10 )
# print(r)
# exit()


def transactions_with_expected_item() -> Iterable[Tuple[Transaction, str]]:
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

def run_real_eval(predictor):
    score = 0
    n_predictions = 0
    for batch in transactions_with_expected_item_batches():

        #print("got batch")

        input_batch = [ trn_with_expected[0] for trn_with_expected in batch ]

        results = predictor.predict_batch(input_batch, 10)
        #print(results)

        for idx, transaction_with_expected_result in enumerate(batch):

            # Get predicted items
            #print(">>>", results)
            predicted_item_labels = results[0][idx]

            n_predictions += 1
            # if n_predictions % 1000 == 0:
            #     print(n_predictions)
            
            expected_item = transaction_with_expected_result[1]
            if expected_item.encode() in predicted_item_labels:
                score += +1

    print("Score: " + str(score) + " of " + str(n_predictions))
    if n_predictions > 0:
        print("Ratio:", str(score / n_predictions))

if __name__ == "__main__":
    predictor = Prediction()
    start = time.time()
    run_real_eval(predictor)
    #cProfile.run('run_real_eval(predictor)', sort=SortKey.CUMULATIVE)
    #cProfile.run('run_real_eval(predictor)', sort=SortKey.TIME)
    end = time.time()
    print("Total time:", end - start)

