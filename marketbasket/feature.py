from marketbasket.jsonutils import read_setting
from marketbasket.labels import Labels
from typing import List, Union
import marketbasket.settings as settings
import tensorflow as tf

class Feature:
    """ A model input/output feature configuration.
        Currently are expected to be string labels
    """

    def __init__(self, config: dict, sequence: bool):
        """ Parse a feature from config file

            Args:
                config: Parsed JSON for this feature
                sequence: True if is a item sequence feature. False if is a transaction feature
        """

        # Feature name
        self.name:str = read_setting(config, 'name', str, Exception("'name' property expected"))

        # If = 0, feature will not be embedded. If > 0, the embedding dimension
        self.embedding_dim:int = read_setting(config, 'embedding_dim', int, 0)

        # Max. number of labels to use as input / predict. All, if == 0.
        self.max_labels = read_setting(config, 'max_labels', int, 0)

        # Belongs to items sequence?
        self.sequence = sequence

        # Label values
        self.labels: Labels = None

    def __repr__(self):
        txt = self.name + ": label "
        txt += "(items sequence feature)" if self.sequence else "(transaction feature)"

        if self.embedding_dim > 0:
            txt += " / embedding_dim: " + str(self.embedding_dim)
        if self.max_labels > 0:
            txt += " / max_labels: " + str(self.max_labels)
        if self.labels:
            txt += " / # labels: " + str(self.labels.length())
        return txt

    def encode_input(self, input:tf.keras.Input, as_multihot = False):
        n_labels = self.labels.length()

        if as_multihot:
            if not self.sequence:
                raise Exception(self.name + " is scalar, it cannot be multihot encoded")
            encoding_layer = tf.keras.layers.Lambda(lambda x: raged_lists_batch_to_multihot(x, n_labels), name="multi_hot_" + 
                self.name)
        elif self.embedding_dim > 0:
            # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
            encoding_layer = tf.keras.layers.Embedding(n_labels + 1, self.embedding_dim, mask_zero=True, name="embedding_" + self.name)
        else:
            # Lambda seems much more fast...
            #return tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=feature.labels.length(), name='one_hot_' + feature.name)
            encoding_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, n_labels), name='one_hot_' + self.name)
        return encoding_layer(input)


@tf.function
def raged_lists_batch_to_multihot(ragged_lists_batch: tf.RaggedTensor, multihot_dim: int) -> tf.Tensor:
    """ Maps a batch of label indices to a batch of multi-hot ones """
    t = ragged_lists_batch.to_tensor(-1) # Default value = -1 -> one_hot will not assign any one
    t = tf.one_hot( t , multihot_dim )
    t = tf.reduce_max( t , axis=1 )
    return t
