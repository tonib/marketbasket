import tensorflow as tf
from labels import Labels
from settings import Settings, ModelType
from gpt import *

def create_model(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:
    if Settings.MODEL_TYPE == ModelType.RNN:
        return create_model_rnn(item_labels, customer_labels)
    elif Settings.MODEL_TYPE == ModelType.DENSE:
        return create_model_non_sequential(item_labels, customer_labels)
    elif Settings.MODEL_TYPE == ModelType.CONVOLUTIONAL:
        # Pending
        return create_model_convolutional(item_labels, customer_labels)
    elif Settings.MODEL_TYPE == ModelType.GPT:
        return create_model_gpt(item_labels, customer_labels)
    else:
        raise Exception("Unknown model type" + Settings.MODEL_TYPE)

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
def pad_sequence( sequences_batch: tf.RaggedTensor) -> tf.Tensor:

    # Avoid sequences larger than sequence_length: Get last sequence_length of each sequence
    sequences_batch = sequences_batch[:,-Settings.SEQUENCE_LENGTH:]
    # Add one to indices, to reserve 0 index for padding
    sequences_batch = sequences_batch + 1
    # Convert to dense, padding zeros to the right
    sequences_batch = sequences_batch.to_tensor(0, shape=[None, Settings.SEQUENCE_LENGTH])
    return sequences_batch

def create_model_rnn(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Input for input items will be a sequence of embeded items
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence(x), name="padded_sequence")(items_input)
    # Embed items sequence:
    n_items = len(item_labels.labels)
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    # TODO: There is a bug in tf2.3: If you set mask_zero=True, GPU and CPU implementations return different values
    # TODO: It seems fixed in tf-nightly. See tf-bugs/gru-bug.py. Try it again in tf2.4
    items_branch = tf.keras.layers.Embedding(n_items + 1, Settings.ITEMS_EMBEDDING_DIM, mask_zero=False)(items_branch)

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, Settings.CUSTOMERS_EMBEDDING_DIM, mask_zero=False)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(Settings.SEQUENCE_LENGTH)(customer_branch)

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
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence(x), name="padded_sequence")(items_input)
    # Embed items sequence:
    n_items = len(item_labels.labels)
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    # TODO: There is a bug in tf2.3: If you set mask_zero=True, GPU and CPU implementations return different values
    # TODO: It seems fixed in tf-nightly. See tf-bugs/gru-bug.py. Try it again in tf2.4
    items_branch = tf.keras.layers.Embedding(n_items + 1, Settings.ITEMS_EMBEDDING_DIM, mask_zero=False)(items_branch)

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, Settings.CUSTOMERS_EMBEDDING_DIM, mask_zero=False)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(Settings.SEQUENCE_LENGTH)(customer_branch)

    # Concatenate embedded customer and items on each timestep
    classification_branch = tf.keras.layers.Concatenate()( [ items_branch , customer_branch ] )

    # Convolution
    convolution_branch = tf.keras.layers.Conv1D(128, 4, activation='relu')(classification_branch)
    # Flatten convolution outputs
    convolution_branch = tf.keras.layers.Flatten()(convolution_branch)

    # RNN
    RNN_LAYER_SIZE = 128
    rnn_layer = tf.keras.layers.GRU(RNN_LAYER_SIZE, return_sequences=True)
    rnn_branch = tf.keras.layers.Bidirectional(rnn_layer)(classification_branch)

    # Dropout. I don't know if this is right for a bidirectional RNN: This can drop a RNN element in forward sequence, but
    # not in backward, and viceversa...
    rnn_branch = tf.keras.layers.Dropout(0.2, noise_shape=[None, 1, RNN_LAYER_SIZE*2])(rnn_branch)
    rnn_branch = tf.keras.layers.Flatten()(rnn_branch)

    # Merge convolution and RNN
    classification_branch = tf.keras.layers.Concatenate()( [ convolution_branch , rnn_branch] )

    classification_branch = tf.keras.layers.Dense(1024, activation='relu')(classification_branch)
    # classification_branch = tf.keras.layers.Dense(1024, activation='relu')(classification_branch)

    # Do the classification (logits)
    classification_branch = tf.keras.layers.Dense(n_items, activation=None)(classification_branch)

    return tf.keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)

##########################################################################################
# GPT
##########################################################################################

def create_model_gpt(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Parameters...?
    num_heads = 2  # Number of attention heads
    feed_forward_dim = 128  # Hidden layer size in feed forward network inside transformer

    # Customer index
    # TODO: Currently unsupported
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)

    # Input for input items will be a sequence of embeded items
    n_items = len(item_labels.labels)
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence(x), name="padded_sequence")(items_input)

    # Items embedding
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    embedding_layer = TokenAndPositionEmbedding(Settings.SEQUENCE_LENGTH, n_items + 1, Settings.ITEMS_EMBEDDING_DIM)
    x = embedding_layer(items_branch)

    # Magic voodoo
    transformer_block = TransformerBlock(Settings.ITEMS_EMBEDDING_DIM, num_heads, feed_forward_dim)
    x = transformer_block(x)
    # transformer_block = TransformerBlock(Settings.ITEMS_EMBEDDING_DIM, num_heads, feed_forward_dim)
    # x = transformer_block(x)

    # Flat output
    # x = tf.keras.layers.Flatten()(x)

    #x = tf.keras.layers.Dense(1024, activation='relu')(x)

    # Classification (logits)
    outputs = layers.Dense(n_items)(x)

    return keras.Model(inputs=[items_input, customer_input], outputs=outputs)
