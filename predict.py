import tensorflow as tf
from labels import Labels
import numpy as np
from operator import itemgetter
from typing import List, Tuple
from settings import Settings
from tensorflow.python.framework.ops import disable_eager_execution
from transaction import Transaction
from model import raged_lists_batch_to_multihot

class Prediction:

    def __init__(self):

        disable_eager_execution()

        self.item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
        self.customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)

        self.model = tf.keras.models.load_model('model/exported_model')
        self.model.summary()

        self.n_items = len(self.item_labels.labels)
        self.n_customers = len(self.customer_labels.labels)

    def _top_predictions(self, result: List[float], item_indices:List[int], n_items_result: int ) -> Tuple[ np.ndarray , np.ndarray ]:

        # "Remove" feeded items indices: Set its probabiblity to negative
        result[ item_indices ] = -1.0
        #print("a", result)
        
        # Get item indices with its highest probability
        top_item_indices = np.argsort( result ) # sort ascending
        top_item_indices = top_item_indices[-n_items_result:] # get top results
        #print( "1", top_item_indices ) 
        top_item_indices = np.flip(top_item_indices) # revert list (most probable first)
        #print( "x", top_item_indices )

        top_probabilities = result[top_item_indices]
        #print( "2", top_probabilities )

        top_labels = self.item_labels.labels[top_item_indices]
        return ( top_labels , top_probabilities )

    def predict_single(self, transaction: Transaction, n_items_result: int) -> List[ Tuple[str, float] ]:
        results = self.predict_batch( [ transaction ] , n_items_result )
        return results[0]

    @tf.function
    def run_prediction(self, batch):
        return self.model(batch)

    #@tf.function
    def predict_batch(self, transactions: List[Transaction], n_items_result: int) -> List:

        # Setup batch
        batch = Transaction.to_net_inputs_batch(transactions, self.item_labels, self.customer_labels)

        # Convert item indices to ragged tensors
        unragged_item_indices = batch[0]
        #batch[0] = tf.ragged.constant(batch[0], dtype=tf.int64)

        #print( "***" , batch )
        # batch_item_indices = tf.ragged.constant(batch_item_indices, dtype=tf.int64)
        # print(">>>", batch_item_indices.to_tensor())
        # if Settings.N_MAX_CUSTOMERS > 0:
        #     batch = ( batch_item_indices , batch_customer_indices )
        # else:
        #     batch = batch_item_indices
        #print(batch)
        #exit()

        #print("**" , batch)
        #   batch = ( np.array(batch[0]) , np.array(batch[1]) )
        batch = ( tf.ragged.constant(batch[0], dtype=tf.int64) , np.array(batch[1]) )
        #print(">>" , batch)
        #print(batch.shape, batch)
        #results = self.run_prediction(batch)
        #results = self.model(batch)
        results = self.model.predict( batch )
        #print(results)

        #print(">>>", results)
        # DO NOT DELETE, TO TEST PYTHON DAMN PERFORMANCE (fake prediction)
        #results = np.random.uniform( size=( len(transactions), len(self.item_labels.labels) ) )

        # Unpack results
        top_predictions = []
        for idx, result in enumerate(results):
            top_predictions.append( self._top_predictions(result, unragged_item_indices[idx], n_items_result) )

        return top_predictions
