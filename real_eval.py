from marketbasket.labels import Labels
from marketbasket.predict import Prediction
from marketbasket.transaction import Transaction
from marketbasket.transactions_file import TransactionsFile
from typing import List, Tuple, Iterable
import cProfile
import re
from pstats import SortKey
import time
import tensorflow as tf
import numpy as np

TEST_BATCH_SIZE = 256

def transactions_with_expected_item() -> Iterable[Tuple[Transaction, str]]:
    with open(TransactionsFile.eval_dataset_path()) as eval_trn_file:
        for line in eval_trn_file:
            transaction = Transaction(line)
            #print("\n\ntransaction:", transaction)

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
    probs_sum = 0.0
    n_predictions = 0
    for batch in transactions_with_expected_item_batches():

        input_batch = [ trn_with_expected[0] for trn_with_expected in batch ]
        #print("input_batch:", input_batch)

        results = predictor.predict_batch(input_batch, 8)
        #print("results:", results)

        for idx, transaction_with_expected_result in enumerate(batch):

            # Get predicted items
            #print(">>>", results)
            predicted_item_labels = results[0][idx]

            n_predictions += 1
            
            expected_item = transaction_with_expected_result[1]
            #print("expected_item", expected_item)

            expected_item_idx = np.where( predicted_item_labels == expected_item.encode() )[0]
            if expected_item_idx.shape[0] > 0:
                prob = results[1][idx][ expected_item_idx[0] ]
                if prob >= 0.01:
                    #print( ">> in" )
                    score += +1
                    probs_sum += results[1][idx][ expected_item_idx[0] ]

    txt_result = "* N. times next item in top eight predictions, with prob. >= 1%: " + str(score) + " of " + str(n_predictions)
    if n_predictions > 0:
        txt_result += " / Ratio:" + str(score / n_predictions)
    print(txt_result)

    txt_result = "* Next item probabilites sum: " + str(probs_sum)
    if n_predictions > 0:
        txt_result += " / Mean: " + str(probs_sum / n_predictions)
    print(txt_result)
    return n_predictions

if __name__ == "__main__":
    predictor = Prediction()
    start = time.time()
    n_predictions = run_real_eval(predictor)
    #cProfile.run('run_real_eval(predictor)', sort=SortKey.CUMULATIVE)
    #cProfile.run('run_real_eval(predictor)', sort=SortKey.TIME)
    end = time.time()
    total_time = end - start
    print("Total time:", total_time)
    print("N. predictions:", n_predictions)
    if n_predictions > 0:
        print("Miliseconds / prediction", (total_time / n_predictions) * 1000.0 )