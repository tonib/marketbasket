import marketbasket.settings as settings
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

# Batch size to run predictions in evaluation
TEST_BATCH_SIZE = 256

settings.settings.features.load_label_files()

def transactions_with_expected_item() -> Iterable[Tuple[Transaction, int]]:
    
    with TransactionsFile(TransactionsFile.eval_dataset_path(), 'r') as eval_trn_file:
        transaction: Transaction
        for transaction in eval_trn_file:
            transaction = transaction.replace_labels_by_indices()

            for idx in range(1, transaction.sequence_length()):
                input_trn = transaction.get_slice(0, idx)
                expected_item_idx = transaction.item_labels[idx]
                yield ( input_trn , expected_item_idx )

def transactions_with_expected_item_batches() -> Iterable[ Tuple[ List[Transaction], List[int] ] ]:
    input_batch = []
    expected_item_indices = []
    for input_trn, expected_item_idx in transactions_with_expected_item():
        input_batch.append( input_trn )
        expected_item_indices.append( expected_item_idx )

        if len( input_batch ) >= TEST_BATCH_SIZE:
            yield input_batch, expected_item_indices
            input_batch = []
            expected_item_indices = []

    # Last batch
    if len(input_batch) > 0:
        yield input_batch, expected_item_indices

def run_real_eval(predictor):
    score = 0
    probs_sum = 0.0
    n_predictions = 0
    n_items_result = 8 # Get only top most probable "n_items_result" predicted items

    # Get input batches with expected items indices
    input_batch: List[Transaction]
    expected_item_indices: List[int]
    for input_batch, expected_item_indices in transactions_with_expected_item_batches():

        # Run prediction over the batch
        top_item_indices, top_probabilities = predictor.predict_raw_batch(input_batch, n_items_result)

        batch_size = len(input_batch) # This may not be TEST_BATCH_SIZE for last batch
        n_predictions += batch_size

        # Check each prediction in batch
        for idx in range(batch_size):

            # Get predicted items
            predicted_item_indices = top_item_indices[idx]
            # Ground truth item            
            expected_item = expected_item_indices[idx]

            # Get index in result where the expected item has been placed
            expected_item_idx = np.where( predicted_item_indices == expected_item )[0]
            if expected_item_idx.shape[0] > 0:
                score += +1
                probs_sum += top_probabilities[idx][ expected_item_idx[0] ]

    txt_result = "* N. times next item in top eight predictions: " + str(score) + " of " + str(n_predictions)
    if n_predictions > 0:
        txt_result += " / Ratio: " + str(score / n_predictions)
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
        print("Milliseconds / prediction", (total_time / n_predictions) * 1000.0 )