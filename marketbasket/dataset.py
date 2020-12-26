from typing import Dict, Tuple
import marketbasket.settings as settings
import tensorflow as tf
from .labels import Labels
import os
import marketbasket.feature
from marketbasket.predict import Prediction
import marketbasket.predict_rating as predict_rating

""" TFRecord dataset setup """

# Constant for output feature name for candidates generation model
OUTPUT_FEATURE_NAME = 'output'

ITEM_TO_RATE = 'item_to_rate'

# Global variable to define what TF type should have read features
_features_to_types:Dict = None

def _setup_feature_keys(rating_model: bool):
    """ Setup features mapping """
    
    rating_model = False

    global _features_to_types
    if _features_to_types:
        # Already defined
        return

    _features_to_types = {}

    # Declare input features
    feature: marketbasket.feature.Feature
    for feature in settings.settings.features:
        if feature.sequence:
            feature_mapped_type = tf.io.RaggedFeature(tf.int64, feature.name, row_splits_dtype=tf.int64)
        else:
            feature_mapped_type = tf.io.FixedLenFeature([], tf.int64)
        _features_to_types[feature.name] = feature_mapped_type

    # Declare output feature
    if rating_model:
        _features_to_types[OUTPUT_FEATURE_NAME] = tf.io.FixedLenFeature([], tf.float32)
        _features_to_types[ITEM_TO_RATE] = tf.io.FixedLenFeature([], tf.int64)
    elif settings.settings.model_type == settings.ModelType.GPT:
        _features_to_types[OUTPUT_FEATURE_NAME] = tf.io.FixedLenFeature([settings.settings.sequence_length], tf.int64)
    else:
        _features_to_types[OUTPUT_FEATURE_NAME] = tf.io.FixedLenFeature([], tf.int64)

@tf.function
def _example_parse_function(proto_batch) -> Tuple:
    """ Convert protobuf Examples batch to TF Tensors batch """

    # Load one batch of examples (MULTIPLE). Load a single example is damn slow
    parsed_features = tf.io.parse_example(proto_batch, _features_to_types)

    output_value = parsed_features[OUTPUT_FEATURE_NAME]
    del parsed_features[OUTPUT_FEATURE_NAME]

    # Return (net inputs, expected output):
    return (parsed_features, output_value)

def _example_parse_function_rating(proto_batch, n_items) -> Tuple:
    """ Convert protobuf Examples batch to TF Tensors batch """

    # Load one batch of examples (MULTIPLE). Load a single example is damn slow
    input_batch = tf.io.parse_example(proto_batch, _features_to_types)

    # This is the expected item to predict
    items_to_predict = input_batch[OUTPUT_FEATURE_NAME]
    del input_batch[OUTPUT_FEATURE_NAME]

    # We will do a negative sampling (1 positive + n_negatives negatives). Repeat inputs
    n_negatives = 10
    n_repeats = 1 + n_negatives
    input_batch = predict_rating.RatingPrediction.repeat_batch_inputs(input_batch,n_repeats)

    # Generate random negatives batch for item indices
    batch_size = tf.shape(items_to_predict)[0]
    negative_items = tf.random.uniform( [batch_size, n_negatives], maxval=n_items, dtype=tf.int64)

    # Add positive items, at the batch begining
    items_to_predict = tf.reshape(items_to_predict, [batch_size, 1]) # [1, 2] -> [[1], [2]]
    items_to_predict = tf.concat( [items_to_predict, negative_items], axis=1) # [[1], [2]], [[10,11] , [20,21]] -> [ [1, 10, 11], [2, 20, 21] ]
    items_to_predict = tf.reshape(items_to_predict, [-1]) # -> [1, 10, 11, 2, 20, 21]
    input_batch[ITEM_TO_RATE] = items_to_predict

    # Assign probabilities
    output = tf.concat( [ [1.0] , tf.zeros([n_negatives]) ], axis=0 ) # -> [ 1.0, 0.0, 0.0 ]
    output = tf.tile( output , [batch_size] ) # -> [ 1.0, 0.0, 0.0, 1.0, 0.0, 0.0 ]

    # Return (net inputs, expected output):
    return (input_batch, output)

def _get_parse_function(rating_model: bool):
    if rating_model:
        n_items = settings.settings.features.items_sequence_feature().labels.length()
        return lambda proto_batch: _example_parse_function_rating(proto_batch, n_items)
    else:
        return _example_parse_function

def train_dataset_file_path(rating_model: bool) -> str:
    """ Returns train dataset file path """
    #return settings.settings.get_data_path( 'dataset_train_candidates.tfrecord' if not rating_model else 'dataset_train_rating.tfrecord' )
    return settings.settings.get_data_path( 'dataset_train_candidates.tfrecord' )

def eval_dataset_file_path(rating_model: bool) -> str:
    """ Returns evaluation dataset file path """
    #return settings.settings.get_data_path( 'dataset_eval_candidates.tfrecord' if not rating_model else 'dataset_eval_rating.tfrecord' )
    return settings.settings.get_data_path( 'dataset_eval_candidates.tfrecord' )

def get_dataset(rating_model: bool, train: bool, debug: bool = False) -> tf.data.Dataset:
    """ Get the train/eval dataset

        Args:
            train: True to load the train dataset. False to load the eval dataset
            debug: True to configure the dataset for debugging (batch size = 1, etc)
    """
    # Map features mappings, if needed
    _setup_feature_keys(rating_model)

    file_path = train_dataset_file_path(rating_model) if train else eval_dataset_file_path(rating_model)

    dataset = tf.data.TFRecordDataset( [ file_path ] )
    dataset = dataset.prefetch(10000)
    if not debug:
        dataset = dataset.shuffle(10000).batch( settings.settings.batch_size )
    else:
        dataset = dataset.batch(1)
    dataset = dataset.map( _get_parse_function(rating_model) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map( _get_parse_function(rating_model) , num_parallel_calls=8)
    return dataset

def n_batches_in_dataset(dataset: tf.data.Dataset) -> int:
    """ Returns the number of batches in the given dataset """
    for n_eval_batches, _ in enumerate(dataset):
        pass
    return n_eval_batches

def write_transaction_to_example(features: dict, writer: tf.io.TFRecordWriter):
    """ Writes an Example in a tfrecord file """
    # Write output with TFRecord format (https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en#creating_a_tftrainexample_message)
    example = tf.train.Example(features=tf.train.Features(feature=features))
    txt_example: str = example.SerializeToString()
    writer.write( txt_example )
