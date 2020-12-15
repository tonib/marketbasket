from marketbasket.settings import settings, ModelType
from marketbasket.transactions_file import TransactionsFile
import tensorflow as tf
import marketbasket.dataset as dataset
from marketbasket.predict import Prediction
from marketbasket.transaction import Transaction
from typing import List
import marketbasket.dataset as dataset
from datetime import datetime

"""
    Generates the train and eval. datasets for rating model
"""

print(datetime.now(), "Process start: Generate rating model datasets")

# Load labels files
settings.features.load_label_files()
settings.print_summary()

BATCH_SIZE = 256

# File to store train samples
train_writer = tf.io.TFRecordWriter( dataset.train_dataset_file_path(True) )
# File to store eval samples
eval_writer = tf.io.TFRecordWriter( dataset.eval_dataset_file_path(True) )

predictor = Prediction()

def process_batch(input_batch: List[Transaction], expected_item_indices: List[int], writer: tf.io.TFRecordWriter):
    """ Generate train/eval sequences for the given transactions batch """
    n_samples = 0
    
    # Run prediction over the batch (get item candidates)
    predicted_item_indices, _, _ = predictor.predict_raw_batch(input_batch, 4)
    
    for transaction_index, transaction in enumerate(input_batch):
        # Ground truth item
        expected_item_index = expected_item_indices[transaction_index]

        # Create the Example from the input transaction
        example_features = transaction.to_example_features()

        expected_found = False
        for candidate_item_index in predicted_item_indices[transaction_index]:
            # Rating ground truth
            if candidate_item_index == expected_item_index:
                expected_found = True
                rating = 1.0
            else:
                rating = 0.0

            example_features[dataset.ITEM_TO_RATE] = tf.train.Feature( int64_list=tf.train.Int64List( value=[candidate_item_index] ) )
            example_features[dataset.OUTPUT_FEATURE_NAME] = tf.train.Feature( float_list=tf.train.FloatList( value=[rating] ) )
            dataset.write_transaction_to_example(example_features, writer)
            n_samples += 1
    
        if not expected_found:
            # Add the expected item
            example_features[dataset.ITEM_TO_RATE] = tf.train.Feature( int64_list=tf.train.Int64List( value=[expected_item_index] ) )
            example_features[dataset.OUTPUT_FEATURE_NAME] = tf.train.Feature( float_list=tf.train.FloatList( value=[1.0] ) )
            dataset.write_transaction_to_example(example_features, writer)
            n_samples += 1

    return n_samples

if settings.model_type == ModelType.GPT:
    raise Exception("GPT model not supported yet")

# Process train transactions
n_samples = 0
with TransactionsFile(TransactionsFile.train_dataset_path(), 'r') as trn_file:
    with tf.io.TFRecordWriter( dataset.train_dataset_file_path(True) ) as writer:
        for input_batch, expected_item_indices in trn_file.transactions_with_expected_item_batches(BATCH_SIZE):
            n_samples += process_batch(input_batch, expected_item_indices, writer)
print("N. train samples", n_samples)

# Process eval transactions
n_samples = 0
with TransactionsFile(TransactionsFile.eval_dataset_path(), 'r') as trn_file:
    with tf.io.TFRecordWriter( dataset.eval_dataset_file_path(True) ) as writer:
        for input_batch, expected_item_indices in trn_file.transactions_with_expected_item_batches(BATCH_SIZE):
            n_samples += process_batch(input_batch, expected_item_indices, writer)
print("N. eval samples", n_samples)

print(datetime.now(), "Process end")