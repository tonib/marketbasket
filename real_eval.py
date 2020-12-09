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
from collections import Counter

# Batch size to run predictions in evaluation
TEST_BATCH_SIZE = 256

# Number of top predicted items to count
N_TOP_PREDICTIONS = 128
OUT_OF_RANKINGS = N_TOP_PREDICTIONS + 1

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

def rank_prediction(predicted_item_indices: np.ndarray, predicted_probabilities:np.ndarray, expected_item:int, 
    prediction_rankings: Counter) -> float:
    # Get index in result where the expected item has been placed
    expected_item_idx = np.where( predicted_item_indices == expected_item )[0]
    if expected_item_idx.shape[0] > 0:
        expected_item_idx = expected_item_idx[0]
        probability = predicted_probabilities[expected_item_idx]
        ranking = expected_item_idx + 1
    else:
        probability = 0.0
        ranking = OUT_OF_RANKINGS
    prediction_rankings[ranking] += 1
    return probability

def print_rankings(prediction_rankings: Counter, n_predictions: int, ranking_top: int):
     # Print results rank
    sum = 0
    for i in range(1, ranking_top + 1):
        if i in prediction_rankings:
            sum += prediction_rankings[i]
    txt_result = "* N. times next item in top " + str(ranking_top) + " predictions: " + str(sum) + " of " + str(n_predictions)
    if n_predictions > 0:
        txt_result += " / Ratio: " + str(sum / n_predictions)
    print(txt_result)

def run_real_eval(predictor):
    probs_sum = 0.0
    n_predictions = 0
    prediction_rankings = Counter()

    # Get input batches with expected items indices
    input_batch: List[Transaction]
    expected_item_indices: List[int]
    for input_batch, expected_item_indices in transactions_with_expected_item_batches():

        # Run prediction over the batch
        top_item_indices, top_probabilities = predictor.predict_raw_batch(input_batch, N_TOP_PREDICTIONS)

        batch_size = len(input_batch) # This may not be TEST_BATCH_SIZE for last batch
        n_predictions += batch_size

        # Check each prediction in batch
        for idx in range(batch_size):

            # Get predicted items and probabilityes
            predicted_item_indices = top_item_indices[idx]
            predicted_probabilities = top_probabilities[idx]

            # Ground truth item            
            expected_item = expected_item_indices[idx]

            # Rank prediction and get item index in prediction
            probability = rank_prediction(predicted_item_indices, predicted_probabilities, expected_item, prediction_rankings)
            probs_sum += probability
            
    # Print rankings
    if n_predictions > 0:
        mean_ranking = sum(ranking * count for ranking, count in prediction_rankings.items()) / n_predictions
        print("Mean ranking:", mean_ranking)
    for rank in [1, 8, 16, 32, 64, 128]:
        print_rankings(prediction_rankings, n_predictions, rank)
    if OUT_OF_RANKINGS in prediction_rankings:
        n_out_of_rank = prediction_rankings[OUT_OF_RANKINGS]
        print("Predictions out of rank: " + str(n_out_of_rank) + " / Ratio: " + str(n_out_of_rank / n_predictions))

    if n_predictions > 0:
        print("Mean probability: " + str(probs_sum / n_predictions))
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