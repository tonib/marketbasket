
import marketbasket.settings as settings
from marketbasket.predict import Prediction
from marketbasket.transaction import Transaction
import marketbasket.dataset as dataset
from typing import List, Tuple, Dict, Union
import numpy as np
import tensorflow as tf

class RatingPrediction(Prediction):
    """ Run and process candidates rating model predictions """

    # Directory in "data" dir where to export the model
    RATING_EXPORTED_MODEL_DIR = 'candidates_exported_model'

    def __init__(self, model:tf.keras.Model = None):
        # Load candidates generation model
        super().__init__()

        if model:
            self._rating_model: tf.keras.Model = model
        else:
            self._rating_model: tf.keras.Model = tf.keras.models.load_model( 
                settings.settings.get_model_path( RatingPrediction.RATING_EXPORTED_MODEL_DIR ) )

    @tf.function
    def repeat_batch_inputs(self, input_batch: Dict[str, Union[tf.Tensor, tf.RaggedTensor]], n_items_result: int) -> Dict[str, Union[tf.Tensor, tf.RaggedTensor]]:
        """ Repeat transaction inputs for each predicted top item """
        result = {}
        for feature_name in input_batch:
            input = input_batch[feature_name]
            if isinstance(input, tf.RaggedTensor):
                # tf.repeat cannot be used with ragged tensors...            
                # ex: input = [ [1] , [2,3] ] , n_items_result = 2
                row_lengths = input.row_lengths() # -> [ 1, 2 ]
                row_lengths = tf.repeat( row_lengths , [n_items_result], axis=0 ) # -> [ 1 , 1 , 2 , 2 ]
                input = tf.tile( input, [1, n_items_result] ) #  -> [ [1, 1] , [2, 3, 2, 3] ]
                input = input.flat_values # -> [ 1, 1 , 2, 3, 2, 3 ]
                input = tf.RaggedTensor.from_row_lengths(input, row_lengths) # -> [ [1], [1], [2, 3], [2, 3] ]
            else:
                input = tf.repeat(input, [n_items_result], axis=0 ) # input = [1 , 2], n_items_result = 2 -> [ 1, 1, 2, 2 ]
            result[feature_name] = input
        return result

    @tf.function
    def calculate_ratings(self, input_batch: Dict[str, Union[tf.Tensor, tf.RaggedTensor]], top_item_indices, n_items_result: int):

        # Repeat transaction inputs for each predicted top item
        print("*** input_batch", input_batch)
        input_batch = self.repeat_batch_inputs(input_batch, n_items_result)

        # Add items to rate to the inputs batch
        #print("*** top_item_indices", top_item_indices)
        items_to_rate = tf.reshape(top_item_indices, [-1])
        input_batch[dataset.ITEM_TO_RATE] = items_to_rate

        print("*** input_batch", input_batch)

        # Get the rating for this transaction. Apply sigmoid, as model generates logits
        new_ratings = self._rating_model(input_batch)
        #new_ratings = tf.nn.sigmoid(new_ratings)

        # Reshape ratings
        print("***new_ratings", input_batch)
        new_ratings = tf.reshape(new_ratings, [-1, n_items_result])
        print("***new_ratings", input_batch)

        # Reorder items/probabilities. Most probable first
        ordered_probs_indices = tf.argsort(new_ratings, direction='DESCENDING')
        print("*** new_ratings.shape", new_ratings.shape)
        print("*** top_item_indices.shape", top_item_indices.shape)
        top_probabilities = tf.gather(new_ratings, ordered_probs_indices, batch_dims=1)
        top_item_indices = tf.gather(top_item_indices, ordered_probs_indices, batch_dims=1)

        print("*** top_item_indices.shape", top_item_indices.shape)
        print("*** top_probabilities.shape", top_probabilities.shape)
        return top_item_indices, top_probabilities

    def predict_raw_batch(self, transactions: List[Transaction], n_items_result: int) -> Tuple[np.ndarray, np.ndarray, List[tf.Tensor]]:
        
        # Get candidate items. Probabilities will be ignored
        top_item_indices, top_probabilities, input_batch = super().predict_raw_batch(transactions, n_items_result)
        top_item_indices, top_probabilities = self.calculate_ratings(input_batch, top_item_indices, n_items_result)
        return top_item_indices.numpy(), top_probabilities.numpy(), input_batch

        # Rate candidates

        # SLOW
        # # Re-compute probabilities with rating model
        # for idx, transaction in enumerate(transactions):
        #     # Ignore empty sequences
        #     if top_probabilities[idx][0] == 0.0:
        #         continue

        #     # Set an input batch for each transaction, with an input for each top item predicted by candidates model
        #     transactions_batch = [transaction] * n_items_result
        #     inputs = self._transactions_to_model_inputs(transactions_batch)
        #     inputs.append( tf.constant(top_item_indices[idx], dtype=tf.int64) )

        #     # Get the rating for this transaction. Apply sigmoid, as model generates logits
        #     new_ratings = self._rating_model(inputs)
        #     new_ratings = tf.nn.sigmoid(new_ratings)

        #     # Set new probabilities
        #     top_probabilities[idx] = tf.reshape( new_ratings.numpy() , -1 )
        # return top_item_indices, top_probabilities