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
            # Special case
            if not self.sequence:
                raise Exception(self.name + " is scalar, it cannot be multihot encoded")
            encoding_layer = tf.keras.layers.Lambda(lambda x: raged_lists_batch_to_multihot(x, n_labels), name="multi_hot_" + 
                self.name)
            return encoding_layer(input)

        # Only embedded sequence features will have mask:
        mask = ( self.sequence and self.embedding_dim > 0 )
        
        if self.sequence:
            # Pad sequence. If masked, add +1 to labels to reserve zero for padding
            layer_name = "pad_zeros_" + self.name
            if mask:
                layer_name = "add_one_" + layer_name
            input = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x, mask), name=layer_name)(input)

        if self.embedding_dim > 0:
            layer_name = "embedding_" + self.name
            if mask:
                # Value zero is reserved for padding
                n_labels += 1
                layer_name = "masked_" + layer_name
            encoding_layer = tf.keras.layers.Embedding(n_labels, self.embedding_dim, mask_zero=mask, name=layer_name)
        else:
            # Lambda seems much more fast...
            #return tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=feature.labels.length(), name='one_hot_' + feature.name)
            encoding_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, n_labels), name='one_hot_' + self.name)
        
        return encoding_layer(input)

###################################################
# TF ENCODING HELPER FUNCTIONS
###################################################

@tf.function
def raged_lists_batch_to_multihot(ragged_lists_batch: tf.RaggedTensor, multihot_dim: int) -> tf.Tensor:
    """ Maps a batch of label indices to a batch of multi-hot ones """
    t = ragged_lists_batch.to_tensor(-1) # Default value = -1 -> one_hot will not assign any one
    t = tf.one_hot( t , multihot_dim )
    t = tf.reduce_max( t , axis=1 )
    return t

@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.bool)])
def pad_sequence_right( sequences_batch: tf.RaggedTensor, mask: bool) -> tf.Tensor:
    """ Pad sequences with zeros on right side """

    # Avoid sequences larger than sequence_length: Get last sequence_length of each sequence
    sequences_batch = sequences_batch[:,-settings.settings.sequence_length:]

    if mask:
        # Add one to indices, to reserve 0 index for padding
        sequences_batch += 1

    # Convert to dense, padding zeros to the right
    sequences_batch = sequences_batch.to_tensor(0, shape=[None, settings.settings.sequence_length])
    return sequences_batch


@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.bool)])
def pad_sequence_left(sequences_batch: tf.RaggedTensor):
    """ Pad sequences with zeros on left side """

    # Truncate rows to have at most `settings.SEQUENCE_LENGTH` items
    sequences_batch = sequences_batch[:,-settings.settings.sequence_length:]

    if mask:
        # Add one to indices, to reserve 0 index for padding
        sequences_batch += 1

    pad_row_lengths = settings.settings.sequence_length - sequences_batch.row_lengths()
    pad_values = tf.zeros( [(settings.settings.sequence_length * sequences_batch.nrows()) - tf.size(sequences_batch, tf.int64)], 
        sequences_batch.dtype)
    padding = tf.RaggedTensor.from_row_lengths(pad_values, pad_row_lengths)
    return tf.concat([padding, sequences_batch], axis=1).to_tensor()
