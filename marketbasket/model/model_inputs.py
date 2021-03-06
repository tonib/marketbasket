import marketbasket.settings as settings
from marketbasket.feature import Feature
import marketbasket.dataset as dataset
from typing import List, Dict, Iterable
import tensorflow as tf

class ModelInputs:
    """ TF model inputs """

    def __init__(self, rating_model: bool):

        self.features = settings.settings.features
        if rating_model:
            # Append item to rate as input transaction feature
            self.features = self.features.rating_model_features()

        # Model inputs. ORDER IS IMPORTANT, as Keras determines inputs by position, and not by name...
        self.inputs: List[tf.keras.Input] = []

        # Inputs by feature
        self.by_feature: Dict[Feature, tf.keras.Input] = {}

        feature: Feature
        for feature in self.features:
            if feature.sequence:
                input = tf.keras.layers.Input(shape=[None], name=feature.name, dtype=tf.int64, ragged=True)
            else:
                input = tf.keras.layers.Input(shape=(), name=feature.name, dtype=tf.int64)
            self.inputs.append(input)
            self.by_feature[feature] = input


    def item_labels_input(self) -> tf.keras.Input:
        """ Get the items sequence input """
        return self.by_feature[self.features.items_sequence_feature()]

    def encode_inputs_set(self, features: Iterable[Feature], concatenate=False, n_repeats=0) -> List:
        """ Encode a set of features """
        encoded_inputs = []
        for feature in features:
            feature_input = self.by_feature[feature]
            encoded_inputs.append( feature.encode_input(feature_input) )

        if len(encoded_inputs) == 0:
            return encoded_inputs
            
        if concatenate and len(encoded_inputs) > 1:
            encoded_inputs = [ tf.keras.layers.Concatenate(axis=1)( encoded_inputs ) ]

        if n_repeats > 0:
            if not concatenate:
                raise Exception("concatenate must to be True to use n_repeats > 0")
            encoded_inputs = [ tf.keras.layers.RepeatVector(n_repeats)(encoded_inputs[0]) ]

        return encoded_inputs

    def get_all_as_sequence(self) -> object:
        # Get encoded sequence inputs
        sequence_inputs = self.encode_inputs_set( self.features.sequence_features() )

        # Get encoded transaction inputs, concatenated, and reapeat for each timestep
        transaction_inputs = self.encode_inputs_set( self.features.transaction_features(), concatenate=True, 
            n_repeats=settings.settings.sequence_length )
        
        # Concatenate sequence and transactions features on each timestep
        return ModelInputs.merge_inputs_set( sequence_inputs + transaction_inputs )
        
    @staticmethod
    def merge_inputs_set(inputs: List) -> object:
        l = len(inputs)
        if l > 1:
            return tf.keras.layers.Concatenate()( inputs )
        elif l == 1:
            return inputs[0]
        else:
            return None
