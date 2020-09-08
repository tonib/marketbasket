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

        self.item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
        self.customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)

        self.model = tf.keras.models.load_model('model/exported_model')
        self.model.summary()

        self.n_items = len(self.item_labels.labels)
        self.n_customers = len(self.customer_labels.labels)

    def _top_predictions(self, result: List[float], item_indices:List[int], n_items_result: int ) -> Tuple[ np.ndarray , np.ndarray ]:
        
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

    # @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64)])
    # def _run_model_prediction(self, batch_item_indices, batch_customer_indices):
    #     return self.model( ( batch_item_indices , batch_customer_indices ) )

    @staticmethod
    @tf.function( input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), 
                                   tf.TensorSpec(shape=[None, None], dtype=tf.float32)] )
    def _remove_input_items_from_prediction(batch_item_indices, result):
        # result is a ragged with input indices for each batch row. Ex [ [0] , [1, 2] ]
        batch_item_indices = batch_item_indices.to_tensor(-1) # -> [ [0,-1] , [1, 2] ]
        #print(batch_item_indices, batch_item_indices.shape[0])

        # Convert batch_item_indices row indices to (row,column) indices
        row_indices = tf.range(0, tf.shape(batch_item_indices)[0], dtype=tf.int64) # -> [ 0, 1 ]
        row_indices = tf.repeat( row_indices, [tf.shape(batch_item_indices)[1]] ) # -> [ 0, 0, 1, 1 ]
        #print(">>>", batch_item_indices)
        batch_item_indices = tf.reshape( batch_item_indices , shape=[-1] ) # -> [ 0, -1, 1, 2 ]
        batch_item_indices = tf.stack( [row_indices, batch_item_indices], axis = 1 ) # -> [ [0,0] , [0,-1], [1,1], [1,2] ]

        # batch_item_indices.to_tensor(-1) added -1's to pad the matrix. Remove these indices
        # Needed according to tf.tensor_scatter_nd_update doc. (it will fail in CPU execution, if there are out of bound indices)
        # Get indices without -1's:
        gather_idxs = tf.where( batch_item_indices[:,1] != -1 ) # -> [[0], [2], [3]]
        batch_item_indices = tf.gather_nd(batch_item_indices, gather_idxs) # -> [ [0,0] , [1,1], [1,2] ]

        # To remove input indices, we will set a probability -1 in their indices
        updates = tf.repeat(-1.0, tf.shape(batch_item_indices)[0]) # -> [ -1, -1, -1 ]

        # Assign -1's to the input indices:
        return tf.tensor_scatter_nd_update( result, batch_item_indices, updates)

    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64)])
    def _run_model_prediction(self, batch_item_indices, batch_customer_indices):
        result = self.model( ( batch_item_indices , batch_customer_indices ) )
        # Set result[ batch_item_indices ] = -1.0:
        return Prediction._remove_input_items_from_prediction( batch_item_indices, result )

    #@tf.function
    def predict_batch(self, transactions: List[Transaction], n_items_result: int) -> List:

        # Setup batch
        batch = Transaction.to_net_inputs_batch(transactions, self.item_labels, self.customer_labels)

        # Convert item indices to ragged tensors
        unragged_item_indices = batch[0]

        # TODO: This will fail if no customer is provided
        batch = ( tf.ragged.constant(batch[0], dtype=tf.int64) , np.array(batch[1]) )
        #print(batch)

        #results = self._run_model_prediction( batch[0] , batch[1] )
        results = self._run_model_prediction( batch[0] , batch[1] )
        results = results.numpy()
        #print(results.shape, results)

        # DO NOT DELETE, TO TEST PYTHON DAMN PERFORMANCE (fake prediction)
        #results = np.random.uniform( size=( len(transactions), len(self.item_labels.labels) ) )

        # Unpack results
        top_predictions = []
        for idx, result in enumerate(results):
            top_predictions.append( self._top_predictions(result, unragged_item_indices[idx], n_items_result) )

        return top_predictions
