from marketbasket.settings import settings, ModelType
from typing import List, Tuple, Dict
import random
from marketbasket.transaction import Transaction
from marketbasket.labels import Labels
import numpy as np
from marketbasket.class_weights import ClassWeights
from datetime import datetime
from marketbasket.transactions_file import TransactionsFile
import marketbasket.dataset as dataset
import tensorflow as tf

"""
    Generates the train and eval. datasets for cantidades generation model
"""

print(datetime.now(), "Process start: Generate candidates model datasets")

# Fix seed to  get reproducible datasets
random.seed(1)

n_train_samples = 0
n_eval_samples = 0

# Load labels files
settings.features.load_label_files()
settings.print_summary()

# File to store train samples
train_writer = tf.io.TFRecordWriter( dataset.train_dataset_file_path(False) )
# File to store eval samples
eval_writer = tf.io.TFRecordWriter( dataset.eval_dataset_file_path(False) )
# File to store eval transactions
eval_trn_file = TransactionsFile(TransactionsFile.eval_dataset_path(), 'w')
# File to store train transactions
train_trn_file = TransactionsFile(TransactionsFile.train_dataset_path(), 'w')

# Number of times each item is used as output (used to weight loss of few used items)
train_item_n_outputs = np.zeros( settings.features.items_sequence_feature().labels.length() , dtype=int)

def write_transaction_to_example(features: dict, eval_transaction: bool):
    """ Writes an Example in a tfrecord file """
    global n_eval_samples, n_train_samples
    if eval_transaction:
        writer = eval_writer
        n_eval_samples += 1
    else:
        writer = train_writer
        n_train_samples += 1

    dataset.write_transaction_to_example(features, writer)

# TODO: Not supported yet
def write_gpt_sample(eval_transaction: bool, input_trn: Transaction, output_items_idx: List[int]):

    # TODO: Explain why
    # Pad output sequence if required
    padding_size = settings.sequence_length - len(output_items_idx)
    if padding_size > 0:
        # Pad with the last output item
        output_items_idx += [output_items_idx[-1]] * padding_size
    elif padding_size < 0:
        output_items_idx = output_items_idx[-settings.sequence_length:]

    # Create the Example, and append expected output 
    example_features = input_trn.to_example_features()
    example_features[dataset.OUTPUT_FEATURE_NAME] = tf.train.Feature( int64_list=tf.train.Int64List( value=output_items_idx ) )

    write_transaction_to_example(example_features, eval_transaction)

# TODO: Not supported yet
def process_trn_gpt_output(eval_transaction: bool, trn_values: Transaction):
    """ Generate train/eval sequences from transaction, for GPT model """
    for item_idx in range(1, trn_values.sequence_length()):

        # Get the input sequence (items before target item, up to "settings.sequence_length" n. items )
        start_idx = item_idx - settings.sequence_length
        if start_idx < 0:
            start_idx = 0
        input_trn = trn_values.get_slice(start_idx, item_idx)

        # Output item indices: The input shifted to left one position, plus target item
        output_items_idx = input_trn.item_labels[1:] + [ trn_values.item_labels[item_idx] ]
        write_gpt_sample(eval_transaction, input_trn, output_items_idx)


def process_trn_single_item_output(eval_transaction: bool, trn_values: Transaction):
    """ Generate train/eval sequences from transaction, for non GPT models """
    # item_idx is the index of item to predict
    for item_idx in range(1, trn_values.sequence_length()):

        # Get the input sequence (items before item to predict, up to "settings.sequence_length" n. items )
        input_trn = trn_values.get_slice(0, item_idx)

        # Output index to predict
        output_item_idx: int = trn_values.item_labels[item_idx]

        # Create the Example, and append expected output 
        example_features = input_trn.to_example_features()
        example_features[dataset.OUTPUT_FEATURE_NAME] = tf.train.Feature( int64_list=tf.train.Int64List( value=[output_item_idx] ) )

        write_transaction_to_example(example_features, eval_transaction)

        if not eval_transaction:
            # Count number of times each item is used as output, for class balancing in train
            train_item_n_outputs[output_item_idx] += 1


def process_transaction(transaction: Transaction):
    """ Generate train/eval sequences for the given transaction """

    # This trn will go to train or evaluation dataset?
    eval_transaction = ( random.random() <= settings.evaluation_ratio )
    if eval_transaction:
        # Store original transaction, for real_eval.py
        eval_trn_file.write( transaction )
    else:
        # For debug
        train_trn_file.write( transaction )

    trn_values = transaction.replace_labels_by_indices()

    # Get sequence items from this transaction
    if settings.model_type == ModelType.GPT:
        # GTP is really different: It generates probabilities for each timestep
        process_trn_gpt_output(eval_transaction, trn_values)
    else:
        process_trn_single_item_output(eval_transaction, trn_values)


# Get transactions
with TransactionsFile(TransactionsFile.top_items_path(), 'r') as trn_file:
    for trn in trn_file:
        process_transaction( trn )
    
train_writer.close()
eval_writer.close()
eval_trn_file.close()
train_trn_file.close()

print("N. train samples", n_train_samples)
print("N. eval samples", n_eval_samples)

if settings.model_type != ModelType.GPT:
    # Save class weights to correct class imbalance
    class_weights = ClassWeights(train_item_n_outputs)
    class_weights.save( ClassWeights.class_weights_path() )

print(datetime.now(), "Process end")
