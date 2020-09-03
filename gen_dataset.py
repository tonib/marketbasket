from typing import List
import itertools
from math import comb
import tensorflow as tf
import random
from settings import Settings

# Calculate number of combinations
# n_combinations = 0
# with open('data/transactions_top_items.txt') as trn_file:
#     for line in trn_file:
#         transaction_items: List[str] = line.split()
#         length = len(transaction_items)
#         for n in range(1,length):
#             c = comb(length, n)
#             if c < 500:
#                 n_combinations += comb(length, n)

# print(n_combinations)

# See https://keunwoochoi.wordpress.com/2020/02/21/tensorflow-parse-tfrecords-tf-io-varlenfeaturetf-string-etc/
# for variable length sequences

n_train_samples = 0
n_eval_samples = 0

with tf.io.TFRecordWriter('data/dataset_train.tfrecord') as train_writer:
    with tf.io.TFRecordWriter('data/dataset_eval.tfrecord') as eval_writer:
        with open('data/transactions_top_items.txt') as trn_file:
            for line in trn_file:
                # Item indices in transaction
                transaction_items: List[int] = [ int(item_index) for item_index in line.split() ]

                # Get each transaction item as output, and get all others as input
                #print(transaction_items)
                for idx in range(0, len(transaction_items)):
                    output: int = int( transaction_items[idx] )
                    input: List[int] = transaction_items[0:idx] + transaction_items[idx+1:]
                    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
                    
                    one_values: List[float] = [1.0] * len(input)
                    features = {
                        'sparse_indices': tf.train.Feature( int64_list=tf.train.Int64List( value=input ) ),
                        'sparse_values' : tf.train.Feature( float_list=tf.train.FloatList( value=one_values ) ),
                        'output_label': tf.train.Feature( int64_list=tf.train.Int64List( value=[output] ) )
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    txt_example: str = example.SerializeToString()
                    if random.random() <= Settings.EVALUATION_RATIO:
                        eval_writer.write( txt_example )
                        n_eval_samples += 1
                    else:
                        train_writer.write( txt_example )
                        n_train_samples += 1
                    # print(input)
                    # print(output)
                    # print(example)

print("N. train samples", n_train_samples)
print("N. eval samples", n_eval_samples)
