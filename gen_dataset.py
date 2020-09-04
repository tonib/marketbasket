from typing import List
import itertools
from math import comb
import tensorflow as tf
import random
from settings import Settings
from transaction import Transaction
from labels import Labels

n_train_samples = 0
n_eval_samples = 0

customer_labels = Labels.load(Settings.CUSTOMER_LABELS_FILE)
item_labels = Labels.load(Settings.ITEM_LABELS_FILE)

def transaction_to_example(customer_idx: int, transaction: Transaction, item_idx: int) -> tf.train.Example:
    output: int = item_labels.label_index( transaction.item_labels[item_idx] )
    input: List[int] = item_labels.labels_indices( transaction.item_labels[0:item_idx] + transaction.item_labels[item_idx+1:] )

    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
    features = {
        'input_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=input ) ),
        'output_item_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[output] ) )
    }
    if Settings.N_MAX_CUSTOMERS > 0:
        features['customer_idx'] = tf.train.Feature( int64_list=tf.train.Int64List( value=[customer_idx] ) )

    return tf.train.Example(features=tf.train.Features(feature=features))

with tf.io.TFRecordWriter(Settings.TRAIN_DATASET_FILE) as train_writer:
    with tf.io.TFRecordWriter(Settings.EVAL_DATASET_FILE) as eval_writer:
        with open('data/transactions_top_items.txt') as trn_file:
            for line in trn_file:
                # Item indices in transaction
                transaction = Transaction(line)
                customer_idx = customer_labels.label_index(transaction.customer_label)

                # Get each transaction item as output, and get all others as input
                for item_idx in range(0, len(transaction.item_labels)):

                    example = transaction_to_example(customer_idx, transaction, item_idx)
                    txt_example: str = example.SerializeToString()
                    if random.random() <= Settings.EVALUATION_RATIO:
                        eval_writer.write( txt_example )
                        n_eval_samples += 1
                    else:
                        train_writer.write( txt_example )
                        n_train_samples += 1

print("N. train samples", n_train_samples)
print("N. eval samples", n_eval_samples)
