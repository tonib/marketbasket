from typing import Dict, Tuple
from .settings import settings, ModelType
import tensorflow as tf
from .labels import Labels
import os
from marketbasket.feature import Feature

""" TFRecord dataset setup """

# Constant for output feature name
OUTPUT_FEATURE_NAME = 'output_item_idx'

# Global variable to define what TF type should have read features
_features_to_types:Dict = None

def _setup_feature_keys():
    """ Setup features mapping """
    
    global _features_to_types
    if _features_to_types:
        # Already defined
        return

    _features_to_types = {}

    # Declare input features
    feature: Feature
    for feature in settings.features:
        if feature.sequence:
            feature_mapped_type = tf.io.RaggedFeature(tf.int64, feature.name, row_splits_dtype=tf.int64)
        else:
            feature_mapped_type = tf.io.FixedLenFeature([], tf.int64)
        _features_to_types[feature.name] = feature_mapped_type

    # Declare output feature
    if settings.model_type == ModelType.GPT:
        _features_to_types[OUTPUT_FEATURE_NAME] = tf.io.FixedLenFeature([settings.sequence_length], tf.int64)
    else:
        _features_to_types[OUTPUT_FEATURE_NAME] = tf.io.FixedLenFeature([], tf.int64)

@tf.function
def _example_parse_function(proto_batch) -> Tuple:
    """ Convert protobuf Examples batch to TF Tensors batch """

    # Load one batch of examples (MULTIPLE). Load a single example is damn slow
    parsed_features = tf.io.parse_example(proto_batch, _features_to_types)

    # Keras inputs are mapped by input POSITION, not by input name (ick, WHY?), so order here is important
    # TODO: We should do this. Probably it will not work
    output_value = parsed_features[OUTPUT_FEATURE_NAME]
    del parsed_features[OUTPUT_FEATURE_NAME]
    # Return (net inputs, expected output):
    return (parsed_features, output_value)


def train_dataset_file_path() -> str:
    """ Returns train dataset file path """
    return settings.get_data_path( 'dataset_train.tfrecord' )

def eval_dataset_file_path() -> str:
    """ Returns evaluation dataset file path """
    return settings.get_data_path( 'dataset_eval.tfrecord' )

def get_dataset(train: bool, debug: bool = False) -> tf.data.Dataset:
    """ Get the train/eval dataset

        Args:
            train: True to load the train dataset. False to load the eval dataset
            debug: True to configure the dataset for debugging (batch size = 1, etc)
    """
    # Map features mappings, if needed
    _setup_feature_keys()

    file_path = train_dataset_file_path() if train else eval_dataset_file_path()

    dataset = tf.data.TFRecordDataset( [ file_path ] )
    dataset = dataset.prefetch(10000)
    if not debug:
        dataset = dataset.shuffle(10000).batch( settings.batch_size )
    else:
        dataset = dataset.batch(1)
    dataset = dataset.map(_example_parse_function , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(_example_parse_function , num_parallel_calls=8)
    return dataset

def n_batches_in_dataset(dataset: tf.data.Dataset) -> int:
    """ Returns the number of batches in the given dataset """
    for n_eval_batches, _ in enumerate(dataset):
        pass
    return n_eval_batches
