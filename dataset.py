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
            'input_items_idx': tf.io.RaggedFeature(tf.int64, 'input_items_idx', row_splits_dtype=tf.int64),
            'customer_idx': tf.io.FixedLenFeature([], tf.int64),
            'output_item_idx': tf.io.FixedLenFeature([], tf.int64),
        }
        DataSet.n_items = len(items_labels.labels)
        DataSet.n_customers = len(customer_labels.labels)

    @staticmethod
    @tf.function
    def example_parse_function(proto_batch):
        # Load one example
        parsed_features = tf.io.parse_example(proto_batch, DataSet.keys_to_features)

        # Keras inputs are mapped by input POSITION, not by input name, so order here is important
        input = ( parsed_features['input_items_idx'] , parsed_features['customer_idx'] )

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
        train_dataset = train_dataset.batch( 2 )
        train_dataset = train_dataset.map( DataSet.example_parse_function )
        return train_dataset
