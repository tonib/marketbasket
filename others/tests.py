import tensorflow as tf
from labels import Labels
from typing import List, Tuple
from transaction import Transaction
from model import pad_sequence # Required to load the model...

class PredictionX(tf.Module):

    def __init__(self, model = None):

        self.item_labels_path = tf.constant( Labels.ITEM_LABELS_FILE )
        self.customer_labels_path = Labels.CUSTOMER_LABELS_FILE

        self.item_indices_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            self.item_labels_path, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
            tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, delimiter=" "), "")

        if model:
            self.model: tf.keras.Model = model
        else:
            self.model: tf.keras.Model = tf.keras.models.load_model('model/exported_model')
            print(">>>", self.model)
            self.model.summary()

    @tf.function( input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                   tf.TensorSpec(shape=[], dtype=tf.int64)] )
    def _top_predictions_tensor(self, results, n_results):

        # Get most probable item indices
        sorted_indices = tf.argsort(results, direction='DESCENDING')
        top_indices = sorted_indices[:,0:n_results]
        top_probabilities = tf.gather(results, top_indices, batch_dims=1)

        # Convert item indices to item labels
        #print("top_indices", top_indices)
        top_item_labels = self.post_process_items(top_indices)
        #top_item_labels = top_indices

        return top_item_labels, top_probabilities

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def post_process_items(self, batch_items_indices: tf.Tensor) -> tf.Tensor:

        # Define reverse lookup tables (tem index -> item string label)
        # Define lookup tables
        # item_indices_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        #     "abc", tf.int64, -1,
        #     tf.string, -2, delimiter=" ", name="item_indices_lookup"), "")

        # # Do lookup
        return self.item_indices_lookup.lookup( tf.cast(batch_items_indices, tf.int64) )

        return batch_items_indices

p = PredictionX()
tf.saved_model.save(p, 'model/serving-model', signatures={ "predicct": p._top_predictions_tensor })
