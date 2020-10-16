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

train_writer = tf.io.TFRecordWriter(DataSet.TRAIN_DATASET_FILE)
eval_writer = tf.io.TFRecordWriter(DataSet.EVAL_DATASET_FILE)

# Number of times each item is used as output (used to weight loss of few used items)
train_item_n_outputs = np.zeros( item_labels.length() , dtype=int)

def enumerate_file_transactions() -> Tuple[ bool , List[int], int ]:
    with open(Transaction.TRANSACTIONS_TOP_ITEMS_PATH) as trn_file:
        with open(Transaction.TRANSACTIONS_EVAL_DATASET_FILE, 'w') as eval_trn_file:
            for line in trn_file:
                transaction = Transaction(line)

                # This trn will go to train or evaluation dataset?
                eval_transaction = ( random.random() <= Settings.EVALUATION_RATIO )
                if eval_transaction:
                    # Store original transaction, for real_eval.py
                    eval_trn_file.write( str(transaction) + '\n' )

                item_indices, customer_idx = transaction.to_net_inputs(item_labels, customer_labels)
                yield ( eval_transaction, item_indices, customer_idx )

def write_transaction_to_example(input_items_idx: List[int], customer_idx: int, output_item_idx: int, eval_transaction: bool):

    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
    features = {
        'input_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=input_items_idx ) ),
        'customer_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[customer_idx] ) ),
        'output_item_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[output_item_idx] ) )
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    txt_example: str = example.SerializeToString()
    if eval_transaction:
        eval_writer.write( txt_example )
        global n_eval_samples
        n_eval_samples += 1
    else:
        train_writer.write( txt_example )
        global n_train_samples
        n_train_samples += 1
        train_item_n_outputs[output_item_idx] += 1

# Get transactions
for eval_transaction, item_indices, customer_idx in enumerate_file_transactions():

    # Get sequence items
    for item_idx in range(1, len(item_indices)):
        output_item_idx: int = item_indices[item_idx]
        input_items_idx: List[int] = item_indices[0:item_idx]
        write_transaction_to_example(input_items_idx, customer_idx, output_item_idx, eval_transaction)
    
train_writer.close()
eval_writer.close()

print("N. train samples", n_train_samples)
print("N. eval samples", n_eval_samples)

# Save class weights to correct class imbalance
class_weights = ClassWeights(train_item_n_outputs)
class_weights.save(ClassWeights.CLASS_WEIGHTS_PATH)
