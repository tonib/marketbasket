import marketbasket.settings as settings
import tensorflow as tf
from marketbasket.labels import Labels
from typing import List, Tuple, Dict
from marketbasket.transaction import Transaction
#from marketbasket.feature import pad_sequence_left, pad_sequence_right # Required to load the model...
import numpy as np

class Prediction(tf.Module):
    """ Run and process candidates generation model predictions """

    # Directory in "models/[MODEL]/" dir where to export the model
    CHECKPOINTS_DIR = 'checkpoints'

    # Directory in "models/[MODEL]/" dir where to export the model
    EXPORTED_MODEL_DIR = 'exported_model'

    def __init__(self, model:tf.keras.Model = None):

        if model:
            self.model: tf.keras.Model = model
        else:
            self.model: tf.keras.Model = tf.keras.models.load_model( settings.settings.get_model_path(False, Prediction.EXPORTED_MODEL_DIR ) )
            #self.model.summary()


    @tf.function( input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                   tf.TensorSpec(shape=[], dtype=tf.int64)] )
    def _top_predictions_tensor(self, results, n_results) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Returns a tuple (items indices, items probabilities) with most probable items, up to "n_results" """
        # Get most probable item indices
        sorted_indices = tf.argsort(results, direction='DESCENDING')
        top_indices = sorted_indices[:,0:n_results]
        top_probabilities = tf.gather(results, top_indices, batch_dims=1)
        return top_indices, top_probabilities


    @staticmethod
    @tf.function( input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), 
                                   tf.TensorSpec(shape=[None, None], dtype=tf.float32)] )
    def _remove_input_items_from_prediction(batch_item_indices, result):
        """ Remove input items from predictions, as we don't want to predict them. It is done setting their
            probabilities to -1

            Args:
                batch_item_indices: Batch input item indices 
                result: Batch predicted probabilities
            Returns: 
                Batch predicted probabilities with input items probs. set to -1
        """
        # batch_item_indices is a ragged with input indices for each batch row. Ex [ [0] , [1, 2] ]
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


    @tf.function
    def _run_model_and_postprocess(self, inputs_batch, item_indices_feature_name, n_results):

        # Run the model
        result = self.model(inputs_batch, training=False)
        # Convert logits to probabilities
        result = tf.nn.softmax(result)

        # Label indices feature values
        batch_item_indices = inputs_batch[item_indices_feature_name]

        if settings.settings.model_type == settings.ModelType.GPT:
            
            # GPT return probabilities for each sequence timestep. 
            # We need the probabilities for the LAST input timested.
            # Batch is a ragged tensors (ex. [[1], [2,3]]), so, get the probabilities for last element position in each 
            # batch sequence:
            indices = batch_item_indices.row_lengths()
            indices -= 1
            #print("indices", indices)
            result = tf.gather(result, indices, batch_dims=1)
            #print("probs", probs)

        # Set result[ batch_item_indices ] = -1.0:
        result = Prediction._remove_input_items_from_prediction( batch_item_indices, result )

        # Get most probable n results
        return self._top_predictions_tensor(result, n_results)


    def _preprocess_transactions(self, transactions: List[Transaction]) -> Tuple[List[Transaction], List[int]]:
        """ Remove unknown items and empty transactions.

            Returns: Non empty transactions, with labels replaced by indices, and indices of empty transactions in
                     the original transactions list
        """
        result = [], empty_sequences_idxs = []
        for idx, trn in enumerate(transactions):
            trn = trn.replace_labels_by_indices()
            # Now, trn item sequence contains -1 for unknown items. Remove sequence feature for these items
            trn = trn.remove_unknown_item_indices()
            if trn.sequence_length() == 0:
                # Empty sequence. If we feed it to the model, it can fail (rnn layers)
                empty_sequences_idxs.append(idx)
            else:
                result.append(trn)

        return result, empty_sequences_idxs

    def _transactions_to_model_inputs(self, transactions: List[Transaction]) -> Dict[str, tf.Tensor]:

        # Prepare dictionary with features names
        inputs_dict = { feature.name : [] for feature in settings.settings.features }

        # Concatenate transaction values for each feature
        for trn in transactions:
            # Use labels indices instead raw values
            for feature in settings.settings.features:
                inputs_dict[feature.name].append( trn[feature.name] )
        
        # To tensor values
        for feature in settings.settings.features:
            if feature.sequence:
                inputs_dict[feature.name] = tf.ragged.constant(inputs_dict[feature.name], dtype=tf.int64)
            else:
                inputs_dict[feature.name] = tf.constant(inputs_dict[feature.name], dtype=tf.int64)

        # Keras inputs are mapped by position, so return result as a list
        #return [ inputs_dict[feature.name] for feature in settings.settings.features ]
        return inputs_dict

    def predict_raw_batch(self, transactions: List[Transaction], n_items_result: int) -> Tuple[np.ndarray, np.ndarray, List[tf.Tensor]]:
        # Transactions to keras inputs
        batch = self._transactions_to_model_inputs(transactions)

        # Run prediction
        if len(batch) > 0:
            top_item_indices, top_probabilities = self._run_model_and_postprocess(batch, settings.settings.features.item_label_feature, 
                n_items_result)
            top_item_indices = top_item_indices.numpy()
            top_probabilities = top_probabilities.numpy()
        else:
            top_item_indices = np.array([], dtype=int)
            top_probabilities = np.array([], dtype=float)

        return top_item_indices, top_probabilities, batch


    def predict_batch(self, transactions: List[Transaction], n_items_result: int, preprocess = True) -> Tuple[np.ndarray, np.ndarray]:

        if preprocess:
            # Convert labels to indices. Remove unknown items and empty sequences
            transactions, empty_sequences_idxs = self._preprocess_transactions(transactions)
        else:
            empty_sequences_idxs = []

        top_item_indices, top_probabilities = self.predict_raw_batch(transactions, n_items_result)

        # Convert item indices to labels
        top_item_labels = settings.settings.features.items_sequence_feature().labels.indices_to_labels(top_item_indices)

        # Insert fake results for empty sequences
        if len(empty_sequences_idxs) > 0:
            empty_labels = np.zeros([n_items_result], dtype=str)
            empty_probs = np.zeros([n_items_result], dtype=float)
            # Inserts cannot be done all at same time, np.insert expects indices to the array before insert, and we don't have them
            for idx in empty_sequences_idxs:
                top_item_labels = np.insert(top_item_labels, idx, empty_labels, axis=0)
                top_probabilities = np.insert(top_probabilities, idx, empty_probs, axis=0)

        return top_item_labels, top_probabilities


    def predict_single(self, transaction: Transaction, n_items_result: int, preprocess = True):
        top_item_labels, top_probabilities = self.predict_batch( [ transaction ] , n_items_result , preprocess )
        return top_item_labels[0] , top_probabilities[0]
