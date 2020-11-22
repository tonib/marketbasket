from typing import List, Tuple
import tensorflow as tf
import random
from marketbasket.settings import settings, ModelType
from marketbasket.transaction import Transaction
from marketbasket.labels import Labels
from marketbasket.dataset import DataSet
import numpy as np
from marketbasket.class_weights import ClassWeights
from datetime import datetime

print(datetime.now(), "Process start: Generate datasets")

# Fix seed to  get reproducible datasets
random.seed(1)

n_train_samples = 0
n_eval_samples = 0

customer_labels = Labels.load(Labels.customer_labels_path())
item_labels = Labels.load(Labels.item_labels_path())

# File to store train samples
train_writer = tf.io.TFRecordWriter( DataSet.train_dataset_file_path() )
# File to store eval samples
eval_writer = tf.io.TFRecordWriter( DataSet.eval_dataset_file_path() )
# File to store eval transactions, to use in real_eval.py
eval_trn_file = open(Transaction.eval_dataset_path(), 'w')

# Number of times each item is used as output (used to weight loss of few used items)
train_item_n_outputs = np.zeros( item_labels.length() , dtype=int)

def write_transaction_to_example(features: dict, eval_transaction: bool):

    global n_eval_samples, n_train_samples
    if eval_transaction:
        writer = eval_writer
        n_eval_samples += 1
    else:
        writer = train_writer
        n_train_samples += 1

    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
    example = tf.train.Example(features=tf.train.Features(feature=features))
    txt_example: str = example.SerializeToString()
    writer.write( txt_example )


def write_gpt_sample(eval_transaction, input_items_idx, customer_idx, output_items_idx):

    # Pad output sequence if required
    padding_size = settings.sequence_length - len(output_items_idx)
    if padding_size > 0:
        # Pad with the last output item
        output_items_idx += [output_items_idx[-1]] * padding_size
    elif padding_size < 0:
        output_items_idx = output_items_idx[-settings.sequence_length:]

    #print( input_items_idx, customer_idx, output_items_idx )

    features = {
        'input_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=input_items_idx ) ),
        'customer_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[customer_idx] ) ),
        'output_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=output_items_idx ) )
    }
    write_transaction_to_example(features, eval_transaction)


def process_trn_gpt_output(eval_transaction, item_indices, customer_idx):
    #print("****", item_indices)

    for item_idx in range(1, len(item_indices)):
        # Input item indices sequence. If input is larger than sequence length, truncate (waste of space)
        input_items_idx: List[int] = item_indices[0:item_idx]
        if len(input_items_idx) > settings.sequence_length:
            input_items_idx = input_items_idx[-settings.sequence_length:]

        # Output item indices: The input shifted to the left one position, plus target item
        output_items_idx = input_items_idx[1:] + [ item_indices[item_idx] ]
        write_gpt_sample(eval_transaction, input_items_idx, customer_idx, output_items_idx)

def process_trn_single_item_output(eval_transaction, item_indices, customer_idx):

    for item_idx in range(1, len(item_indices)):

        # Output index to predict
        output_item_idx: int = item_indices[item_idx]

        # Input item indices sequence. If input is larger than sequence length, truncate (waste of space)
        input_items_idx: List[int] = item_indices[0:item_idx]
        if len(input_items_idx) > settings.sequence_length:
            input_items_idx = input_items_idx[-settings.sequence_length:]

        #print( input_items_idx, customer_idx, output_item_idx )
        features = {
            'input_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=input_items_idx ) ),
            'customer_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[customer_idx] ) ),
            'output_item_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[output_item_idx] ) )
        }
        write_transaction_to_example(features, eval_transaction)

        if not eval_transaction:
            # Count number of times each item is used as output, for class balancing in train
            train_item_n_outputs[output_item_idx] += 1


def process_transaction(transaction: Transaction):

    item_indices, customer_idx = transaction.to_net_inputs(item_labels, customer_labels)

    # This trn will go to train or evaluation dataset?
    eval_transaction = ( random.random() <= settings.evaluation_ratio )
    if eval_transaction:
        # Store original transaction, for real_eval.py
        eval_trn_file.write( str(transaction) + '\n' )

    # Get sequence items from this transaction
    if settings.model_type == ModelType.GPT:
        # Add sequence and its reverse (finally not reverse)
        process_trn_gpt_output(eval_transaction, item_indices, customer_idx)
        # if not eval_transaction:
        #     process_trn_gpt_output(eval_transaction, list(reversed(item_indices)), customer_idx)
    else:
        process_trn_single_item_output(eval_transaction, item_indices, customer_idx)


# Get transactions
with open(Transaction.top_items_path()) as trn_file:
    for line in trn_file:
        process_transaction( Transaction(line) )
    
train_writer.close()
eval_writer.close()
eval_trn_file.close()

print("N. train samples", n_train_samples)
print("N. eval samples", n_eval_samples)

# Save class weights to correct class imbalance
class_weights = ClassWeights(train_item_n_outputs)
class_weights.save( ClassWeights.class_weights_path() )

print(datetime.now(), "Process end")
