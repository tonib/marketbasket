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

def write_transaction_to_example(input_items_idx: List[int], customer_idx: int, output_item_idx: int, gpt_output: List[int],
writer: tf.io.TFRecordWriter):

    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
    features = {
        'input_items_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=input_items_idx ) ),
        'customer_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[customer_idx] ) ),
        'output_item_idx': tf.train.Feature( int64_list=tf.train.Int64List( value=[output_item_idx] ) ),
        'gpt_output': tf.train.Feature( int64_list=tf.train.Int64List( value=gpt_output ) )
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    txt_example: str = example.SerializeToString()
    writer.write( txt_example )

def get_gpt_output(input_items_idx: List[int], output_item_idx: int) -> List[int]:
    # GPT predicts the entire sequence, one index shifted to left. 
    # Example with seq. length = 5 -> input: [a, b, c , padding, padding], prediction: a, b, c, D, padding]
    # Other example: [a, b, c, d, e] -> [b, c, d, e, F]
    
    gpt_output = input_items_idx + [output_item_idx]

    # zero is reserved for padding, so add 1 to all indices
    gpt_output = [x + 1 for x in gpt_output]

    padding_size = Settings.SEQUENCE_LENGTH - len(gpt_output)
    if padding_size > 0:
        gpt_output += [0] * padding_size
    elif padding_size < 0:
        gpt_output = gpt_output[-Settings.SEQUENCE_LENGTH:]
    
    #print( input_items_idx , output_item_idx , gpt_output)
    return gpt_output

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

        # Output index to predict
        output_item_idx: int = item_indices[item_idx]

        # Input item indices sequence. If input is larger than sequence length, truncate (waste of space)
        input_items_idx: List[int] = item_indices[0:item_idx]
        if len(input_items_idx) > Settings.SEQUENCE_LENGTH:
            input_items_idx = input_items_idx[-Settings.SEQUENCE_LENGTH:]

        # GPT output to predict
        gpt_output = get_gpt_output(input_items_idx, output_item_idx)

        write_transaction_to_example(input_items_idx, customer_idx, output_item_idx, gpt_output, writer)

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
