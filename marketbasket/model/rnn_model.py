from marketbasket.settings import settings
import tensorflow as tf
from .model_inputs import ModelInputs

@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64)])
def pad_sequence_right( sequences_batch: tf.RaggedTensor) -> tf.Tensor:
    """ Pad sequences with zeros on right side """

    # Avoid sequences larger than sequence_length: Get last sequence_length of each sequence
    sequences_batch = sequences_batch[:,-settings.sequence_length:]
    # Add one to indices, to reserve 0 index for padding
    sequences_batch = sequences_batch + 1
    # Convert to dense, padding zeros to the right
    sequences_batch = sequences_batch.to_tensor(0, shape=[None, settings.sequence_length])
    return sequences_batch


@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64)])
def pad_sequence_left(sequences_batch: tf.RaggedTensor):
    """ Pad sequences with zeros on left side """

    sequences_batch = sequences_batch[:,-settings.sequence_length:]  # Truncate rows to have at most `settings.SEQUENCE_LENGTH` items

    # Add one to indices, to reserve 0 index for padding
    sequences_batch = sequences_batch + 1

    pad_row_lengths = settings.sequence_length - sequences_batch.row_lengths()
    pad_values = tf.zeros( [(settings.sequence_length * sequences_batch.nrows()) - tf.size(sequences_batch, tf.int64)] , sequences_batch.dtype)
    padding = tf.RaggedTensor.from_row_lengths(pad_values, pad_row_lengths)
    return tf.concat([padding, sequences_batch], axis=1).to_tensor()


def create_model_rnn(inputs: ModelInputs) -> tf.keras.Model:

    # Input for input items will be a sequence of embedded items
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)
    # Embed items sequence:
    n_items = len(item_labels.labels)
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    # TODO: There is a bug in tf2.3: If you set mask_zero=True, GPU and CPU implementations return different values
    # TODO: It seems fixed in tf-nightly. See tf-bugs/gru-bug.py. Try it again in tf2.4
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.items_embedding_dim, mask_zero=False)(items_branch)

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.customers_embedding_dim, mask_zero=False)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.sequence_length)(customer_branch)

    # Concatenate embedded customer and items on each timestep
    classification_branch = tf.keras.layers.Concatenate()( [ items_branch , customer_branch ] )

    # Define RNN
    RNN_LAYER_SIZE = 128
    rnn_layer = tf.keras.layers.GRU(RNN_LAYER_SIZE, return_sequences=True)
    classification_branch = tf.keras.layers.Bidirectional(rnn_layer)(classification_branch)

    # Dropout. I don't know if this is right for a bidirectional RNN: This can drop a RNN element in forward sequence, but
    # not in backward, and viceversa...
    classification_branch = tf.keras.layers.Dropout(0.2, noise_shape=[None, 1, RNN_LAYER_SIZE*2])(classification_branch)

    # Flatten RNN outputs
    classification_branch = tf.keras.layers.Flatten()(classification_branch)

    # Do the classification (Logits)
    classification_branch = tf.keras.layers.Dense(n_items, activation=None)(classification_branch)

    return tf.keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)
