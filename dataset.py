import tensorflow as tf
from settings import Settings
from labels import Labels

class DataSet:

    # Train dataset file path
    TRAIN_DATASET_FILE = 'data/dataset_train.tfrecord'

    # Train dataset file path
    EVAL_DATASET_FILE = 'data/dataset_eval.tfrecord'

    @staticmethod
    def setup_feature_keys(items_labels: Labels, customer_labels: Labels):
        # Feature mappings for training
        DataSet.keys_to_features = {
            'input_items_idx': tf.io.RaggedFeature(tf.int64, 'input_items_idx'),
            'output_item_idx': tf.io.FixedLenFeature([], tf.int64)
        }
        if Settings.N_MAX_CUSTOMERS > 0:
            DataSet.keys_to_features['customer_idx'] = tf.io.FixedLenFeature([], tf.int64)

        DataSet.n_items = len(items_labels.labels)
        DataSet.n_customers = len(customer_labels.labels)

    @staticmethod
    @tf.function
    def raged_lists_batch_to_multihot(ragged_lists_batch, multihot_dim):
        t = tf.one_hot( ragged_lists_batch , multihot_dim )
        t = tf.reduce_max( t , axis=1 )
        return t

    @staticmethod
    @tf.function
    def example_parse_function(proto_batch):
        # Load one example
        parsed_features = tf.io.parse_example(proto_batch, DataSet.keys_to_features)

        # Map batch of list of item indices to a batch of multi-hot arrays
        # Ex [ [1, 2] , [2, 3] ] -> [ [ 0 , 1 , 1 , 0 ] , [ 0 , 0 , 1 , 1 ] ]
        items_input = parsed_features['input_items_idx']
        items_input = DataSet.raged_lists_batch_to_multihot( items_input , DataSet.n_items )

        if Settings.N_MAX_CUSTOMERS > 0:
            input = { 'input_items_idx': items_input , 'customer_idx': parsed_features['customer_idx'] }
        else:
            input = items_input

        # Return tuple (net input, expected output)
        return input, parsed_features['output_item_idx']
            
    @staticmethod
    def n_eval_batches() -> int:
        # We need the batches number in evaluation dataset, so here is:
        # (This will be executed in eager mode)
        train_dataset = tf.data.TFRecordDataset( [ DataSet.EVAL_DATASET_FILE ] )
        train_dataset = train_dataset.batch( Settings.BATCH_SIZE )
        for n_eval_batches, _ in enumerate(train_dataset):
            pass
        # print(f'Number of elements: {n_eval_batches}')
        return n_eval_batches

    @staticmethod
    def load_train_dataset() -> tf.data.Dataset:
        train_dataset = tf.data.TFRecordDataset( [ DataSet.TRAIN_DATASET_FILE ] )
        train_dataset = train_dataset.prefetch(10000)
        train_dataset = train_dataset.shuffle(10000).batch( Settings.BATCH_SIZE )
        train_dataset = train_dataset.map( DataSet.example_parse_function , num_parallel_calls=8 )
        return train_dataset

    @staticmethod
    def load_eval_dataset() -> tf.data.Dataset:
        eval_dataset = tf.data.TFRecordDataset( [ DataSet.EVAL_DATASET_FILE ] )
        eval_dataset = eval_dataset.prefetch(10000)
        eval_dataset = eval_dataset.batch( Settings.BATCH_SIZE )
        eval_dataset = eval_dataset.map( DataSet.example_parse_function )
        return eval_dataset

    @staticmethod
    def load_debug_train_dataset() -> tf.data.Dataset:
        train_dataset = tf.data.TFRecordDataset( [ DataSet.TRAIN_DATASET_FILE ] )
        train_dataset = train_dataset.batch( 1 )
        train_dataset = train_dataset.map( DataSet.example_parse_function )
        return train_dataset
