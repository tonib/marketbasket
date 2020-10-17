from typing import List, Tuple
import tensorflow as tf
import random
from settings import Settings
from transaction import Transaction
from labels import Labels
from dataset import DataSet
import numpy as np
from class_weights import ClassWeights

# Fix seed to  get reproducible datasets
random.seed(1)

n_train_samples = 0
n_eval_samples = 0

customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)
item_labels = Labels.load(Labels.ITEM_LABELS_FILE)

# File to store train samples
train_writer = tf.io.TFRecordWriter(DataSet.TRAIN_DATASET_FILE)
# File to store eval samples
eval_writer = tf.io.TFRecordWriter(DataSet.EVAL_DATASET_FILE)
# File to store eval transactions, to use in real_eval.py
eval_trn_file = open(Transaction.TRANSACTIONS_EVAL_DATASET_FILE, 'w')

# Number of times each item is used as output (used to weight loss of few used items)
train_item_n_outputs = np.zeros( item_labels.length() , dtype=int)

def write_transaction_to_example(input_items_idx: List[int], customer_idx: int, output_item_idx: int, writer: tf.io.TFRecordWriter):

    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
    features = {
        'input_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=input_items_idx ) ),
        'customer_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[customer_idx] ) ),
        'output_item_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[output_item_idx] ) )
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    txt_example: str = example.SerializeToString()
    writer.write( txt_example )


def process_transaction(transaction: Transaction):

    item_indices, customer_idx = transaction.to_net_inputs(item_labels, customer_labels)

    # This trn will go to train or evaluation dataset?
    eval_transaction = ( random.random() <= Settings.EVALUATION_RATIO )
    if eval_transaction:
        # Store original transaction, for real_eval.py
        eval_trn_file.write( str(transaction) + '\n' )
        # Generated sequences will go to evaluation
        writer = eval_writer
    else:
        # Sequences will go to train
        writer = train_writer

    global n_eval_samples, n_train_samples

    # Get sequence items from this transaction
    for item_idx in range(1, len(item_indices)):
        output_item_idx: int = item_indices[item_idx]
        input_items_idx: List[int] = item_indices[0:item_idx]
        write_transaction_to_example(input_items_idx, customer_idx, output_item_idx, writer)

        if eval_transaction:
            n_eval_samples += 1
        else:
            n_train_samples += 1
            # Count number of times each item is used as output, for class balancing in train
            train_item_n_outputs[output_item_idx] += 1


# Get transactions
with open(Transaction.TRANSACTIONS_TOP_ITEMS_PATH) as trn_file:
    for line in trn_file:
        process_transaction( Transaction(line) )
    
train_writer.close()
eval_writer.close()
eval_trn_file.close()

print("N. train samples", n_train_samples)
print("N. eval samples", n_eval_samples)

# Save class weights to correct class imbalance
class_weights = ClassWeights(train_item_n_outputs)
class_weights.save(ClassWeights.CLASS_WEIGHTS_PATH)
