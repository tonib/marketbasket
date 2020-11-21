import tensorflow as tf
from labels import Labels
from settings import settings, ModelType
from gpt import *

def create_model(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:
    if settings.MODEL_TYPE == ModelType.RNN:
        return create_model_rnn(item_labels, customer_labels)
    elif settings.MODEL_TYPE == ModelType.DENSE:
        return create_model_non_sequential(item_labels, customer_labels)
    elif settings.MODEL_TYPE == ModelType.CONVOLUTIONAL:
        # Pending
        return create_model_convolutional_v2(item_labels, customer_labels)
    elif settings.MODEL_TYPE == ModelType.GPT:
        return create_model_gpt_raw(item_labels, customer_labels)
    else:
        raise Exception("Unknown model type" + settings.MODEL_TYPE)

##########################################################################################
# DENSE
##########################################################################################

@tf.function
def raged_lists_batch_to_multihot(ragged_lists_batch: tf.RaggedTensor, multihot_dim: int) -> tf.Tensor:
    t = ragged_lists_batch.to_tensor(-1) # Default value = -1 -> one_hot will not assign any one
    t = tf.one_hot( t , multihot_dim )
    t = tf.reduce_max( t , axis=1 )
    return t

def create_model_non_sequential(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    n_items = len(item_labels.labels)
    n_customers = len(customer_labels.labels)

    # Input for input items will be a multihot array
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    encoded_items_layer = tf.keras.layers.Lambda(lambda x: raged_lists_batch_to_multihot(x, n_items), name="multi_hot_items_encoding")
    encoded_items = encoded_items_layer(items_input)

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    # Conver to one-hot
    customer_encoder_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, n_customers), name='one_hot_customer_encoding')
    customer_input_encoded = customer_encoder_layer(customer_input)

    input_layer = tf.keras.layers.Concatenate(axis=1)( [ encoded_items , customer_input_encoded ] )
    
    x = tf.keras.layers.Dense(1024, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Classification (logits)
    x = tf.keras.layers.Dense(n_items, activation=None)(x)

    return tf.keras.Model(inputs=[ items_input , customer_input ], outputs=x)
    
##########################################################################################
# RNN
##########################################################################################

@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64)])
def pad_sequence_right( sequences_batch: tf.RaggedTensor) -> tf.Tensor:
    """ Pad sequences with zeros on right side """

    # Avoid sequences larger than sequence_length: Get last sequence_length of each sequence
    sequences_batch = sequences_batch[:,-settings.SEQUENCE_LENGTH:]
    # Add one to indices, to reserve 0 index for padding
    sequences_batch = sequences_batch + 1
    # Convert to dense, padding zeros to the right
    sequences_batch = sequences_batch.to_tensor(0, shape=[None, settings.SEQUENCE_LENGTH])
    return sequences_batch

@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64)])
def pad_sequence_left(sequences_batch: tf.RaggedTensor):
    """ Pad sequences with zeros on left side """

    sequences_batch = sequences_batch[:,-settings.SEQUENCE_LENGTH:]  # Truncate rows to have at most `settings.SEQUENCE_LENGTH` items

    # Add one to indices, to reserve 0 index for padding
    sequences_batch = sequences_batch + 1

    pad_row_lengths = settings.SEQUENCE_LENGTH - sequences_batch.row_lengths()
    pad_values = tf.zeros( [(settings.SEQUENCE_LENGTH * sequences_batch.nrows()) - tf.size(sequences_batch, tf.int64)] , sequences_batch.dtype)
    padding = tf.RaggedTensor.from_row_lengths(pad_values, pad_row_lengths)
    return tf.concat([padding, sequences_batch], axis=1).to_tensor()


def create_model_rnn(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Input for input items will be a sequence of embeded items
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)
    # Embed items sequence:
    n_items = len(item_labels.labels)
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    # TODO: There is a bug in tf2.3: If you set mask_zero=True, GPU and CPU implementations return different values
    # TODO: It seems fixed in tf-nightly. See tf-bugs/gru-bug.py. Try it again in tf2.4
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.ITEMS_EMBEDDING_DIM, mask_zero=False)(items_branch)

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.CUSTOMERS_EMBEDDING_DIM, mask_zero=False)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.SEQUENCE_LENGTH)(customer_branch)

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

##########################################################################################
# CONVOLUTIONAL (NOT REALLY)
##########################################################################################

def create_model_convolutional(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Not really a convolutional. It's a RNN + Convolutional ensemble

    # Input for input items will be a sequence of embeded items
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)
    # Embed items sequence:
    n_items = len(item_labels.labels)
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    # TODO: There is a bug in tf2.3: If you set mask_zero=True, GPU and CPU implementations return different values
    # TODO: It seems fixed in tf-nightly. See tf-bugs/gru-bug.py. Try it again in tf2.4
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.ITEMS_EMBEDDING_DIM, mask_zero=True)(items_branch)

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.CUSTOMERS_EMBEDDING_DIM, mask_zero=False)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.SEQUENCE_LENGTH)(customer_branch)

    # Concatenate embedded customer and items on each timestep
    classification_branch = tf.keras.layers.Concatenate()( [ items_branch , customer_branch ] )

    # Convolution
    convolution_branch = tf.keras.layers.Conv1D(128, 4, activation='relu')(classification_branch)
    # Flatten convolution outputs
    convolution_branch = tf.keras.layers.Flatten()(convolution_branch)

    # RNN
    RNN_LAYER_SIZE = 128
    rnn_layer = tf.keras.layers.GRU(RNN_LAYER_SIZE, return_sequences=False)
    rnn_branch = tf.keras.layers.Bidirectional(rnn_layer)(classification_branch)

    # Dropout. I don't know if this is right for a bidirectional RNN: This can drop a RNN element in forward sequence, but
    # not in backward, and viceversa...
    rnn_branch = tf.keras.layers.Dropout(0.2)(rnn_branch)

    # Merge convolution and RNN
    classification_branch = tf.keras.layers.Concatenate()( [ convolution_branch , rnn_branch] )

    classification_branch = tf.keras.layers.Dense(512, activation='relu')(classification_branch)
    # classification_branch = tf.keras.layers.Dense(1024, activation='relu')(classification_branch)

    # Do the classification (logits)
    classification_branch = tf.keras.layers.Dense(n_items, activation=None)(classification_branch)

    return tf.keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)


# Same as create_model_convolutional, with two stacked conv1d
def create_model_convolutional_v2(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Not really a convolutional. It's a RNN + Convolutional ensemble

    # Input for input items will be a sequence of embeded items
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)
    # Embed items sequence:
    n_items = len(item_labels.labels)
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    # TODO: There is a bug in tf2.3: If you set mask_zero=True, GPU and CPU implementations return different values
    # TODO: It seems fixed in tf-nightly. See tf-bugs/gru-bug.py. Try it again in tf2.4
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.ITEMS_EMBEDDING_DIM, mask_zero=True)(items_branch)

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.CUSTOMERS_EMBEDDING_DIM, mask_zero=False)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.SEQUENCE_LENGTH)(customer_branch)

    # Concatenate embedded customer and items on each timestep
    classification_branch = tf.keras.layers.Concatenate()( [ items_branch , customer_branch ] )

    # Convolution
    convolution_branch = tf.keras.layers.Conv1D(128, 4, activation='relu')(classification_branch)
    convolution_branch = tf.keras.layers.Conv1D(128, 4, activation='relu')(convolution_branch)

    # Flatten convolution outputs
    convolution_branch = tf.keras.layers.Flatten()(convolution_branch)

    # RNN
    RNN_LAYER_SIZE = 128
    rnn_layer = tf.keras.layers.GRU(RNN_LAYER_SIZE, return_sequences=False)
    rnn_branch = tf.keras.layers.Bidirectional(rnn_layer)(classification_branch)

    # Dropout. I don't know if this is right for a bidirectional RNN: This can drop a RNN element in forward sequence, but
    # not in backward, and viceversa...
    rnn_branch = tf.keras.layers.Dropout(0.2)(rnn_branch)

    # Merge convolution and RNN
    classification_branch = tf.keras.layers.Concatenate()( [ convolution_branch , rnn_branch] )

    classification_branch = tf.keras.layers.Dense(512, activation='relu')(classification_branch)

    # Do the classification (logits)
    classification_branch = tf.keras.layers.Dense(n_items, activation=None)(classification_branch)

    return tf.keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)


##########################################################################################
# GPT
##########################################################################################

def create_model_gpt_with_conv(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.CUSTOMERS_EMBEDDING_DIM)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.SEQUENCE_LENGTH)(customer_branch)

    # Input for input items will be a sequence of embeded items
    n_items = len(item_labels.labels)
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)

    # Items embedding
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.ITEMS_EMBEDDING_DIM)(items_branch)

    # Append positional encoding for each item sequence index
    transformer_branch = AddPositionEmbedding(settings.SEQUENCE_LENGTH, settings.ITEMS_EMBEDDING_DIM)(items_branch)
    # Concatenate embedded customer and items on each timestep
    transformer_branch = tf.keras.layers.Concatenate()( [ transformer_branch , customer_branch ] )
    # Transformer decoder
    num_heads = 8  # Number of attention heads
    feed_forward_dim = 128  # Hidden layer size in feed forward network inside transformer
    transformer_branch = TransformerBlock(settings.ITEMS_EMBEDDING_DIM + settings.CUSTOMERS_EMBEDDING_DIM, num_heads, feed_forward_dim)(transformer_branch)

    # Add customer to input (without position encoding)
    convolution_branch = tf.keras.layers.Concatenate()( [ items_branch , customer_branch ] )
    #print(">>>> convolution_branch", convolution_branch)
    # Convolution
    convolution_branch = tf.keras.layers.Conv1D(64, 4, activation='relu')(convolution_branch)
    # Flatten convolution outputs
    convolution_branch = tf.keras.layers.Flatten()(convolution_branch)
    # Repeat for each timestep
    convolution_branch = tf.keras.layers.RepeatVector(settings.SEQUENCE_LENGTH)(convolution_branch)

    # Concatenate transformer output and convolution output on each timestep
    classification_branch = tf.keras.layers.Concatenate()( [ transformer_branch , convolution_branch ] )

    # Process transformer output and context 
    classification_branch = tf.keras.layers.Dense(512, activation='relu')(classification_branch)

    # Classification (logits)
    classification_branch = layers.Dense(n_items)(classification_branch)

    return keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)


def create_model_gpt_raw(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.CUSTOMERS_EMBEDDING_DIM)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.SEQUENCE_LENGTH)(customer_branch)

    # Input for input items will be a sequence of embedded items
    n_items = len(item_labels.labels)
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)

    # Items embedding
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.ITEMS_EMBEDDING_DIM)(items_branch)

    # Append positional encoding for each item sequence index
    transformer_branch = AddPositionEmbedding(settings.SEQUENCE_LENGTH, settings.ITEMS_EMBEDDING_DIM)(items_branch)
    # Concatenate embedded customer and items on each timestep
    transformer_branch = tf.keras.layers.Concatenate()( [ transformer_branch , customer_branch ] )
    # Transformer decoder
    num_heads = 8  # Number of attention heads
    feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer
    transformer_branch = TransformerBlock(settings.ITEMS_EMBEDDING_DIM + settings.CUSTOMERS_EMBEDDING_DIM, num_heads, feed_forward_dim)(transformer_branch)

    # Classification (logits)
    classification_branch = layers.Dense(n_items)(transformer_branch)

    return keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)
