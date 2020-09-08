from typing import List
import itertools
from math import comb
import tensorflow as tf
import random
from settings import Settings
from transaction import Transaction
from labels import Labels
from dataset import DataSet

n_train_samples = 0
n_eval_samples = 0

customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)
item_labels = Labels.load(Labels.ITEM_LABELS_FILE)

def transaction_to_example(customer_idx: int, transaction: Transaction, item_idx: int) -> tf.train.Example:
    output: int = item_labels.label_index( transaction.item_labels[item_idx] )
    input: List[int] = item_labels.labels_indices( transaction.item_labels[0:item_idx] + transaction.item_labels[item_idx+1:] )

    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
    features = {
        'input_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=input ) ),
        'customer_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[customer_idx] ) ),
        'output_item_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[output] ) )
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

with tf.io.TFRecordWriter(DataSet.TRAIN_DATASET_FILE) as train_writer:
    with tf.io.TFRecordWriter(DataSet.EVAL_DATASET_FILE) as eval_writer:
        with open(Transaction.TRANSACTIONS_EVAL_DATASET_FILE, 'w') as eval_trn_file:
            with open(Transaction.TRANSACTIONS_TOP_ITEMS_PATH) as trn_file:
                for line in trn_file:
                    # Item indices in transaction
                    transaction = Transaction(line)
                    customer_idx = customer_labels.label_index(transaction.customer_label)

                    # This trn will go to train or evaluation dataset?
                    eval_transaction = ( random.random() <= Settings.EVALUATION_RATIO )
                    if eval_transaction:
                        # Store original transaction, for real_eval.py
                        eval_trn_file.write( str(transaction) + '\n' )

                    # Get each transaction item as output, and get all others as input
                    for item_idx in range(0, len(transaction.item_labels)):
                        example = transaction_to_example(customer_idx, transaction, item_idx)
                        txt_example: str = example.SerializeToString()
                        if eval_transaction:
                            eval_writer.write( txt_example )
                            n_eval_samples += 1
                        else:
                            train_writer.write( txt_example )
                            n_train_samples += 1

print("N. train samples", n_train_samples)
print("N. eval samples", n_eval_samples)
