import tensorflow as tf
from labels import Labels
from settings import Settings

def create_model(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:
    if Settings.SEQUENTIAL:
        return create_model_sequential(item_labels, customer_labels)
    else:
        return create_model_non_sequential(item_labels, customer_labels)

##########################################################################################
# NON SEQUENTIAL
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
    
    x = tf.keras.layers.Dense(256)(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(n_items, activation='softmax')(x)

    return tf.keras.Model(inputs=[ items_input , customer_input ], outputs=x)
    
##########################################################################################
# SEQUENTIAL
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


def create_model_sequential(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:
    
    # SEE https://github.com/tensorflow/tensorflow/issues/36508

    # Input for input items will be a sequence of embeded items
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence(x), name="padded_sequence")(items_input)
    # Embbed items sequence:
    n_items = len(item_labels.labels)
    # +1 in "n_items + 1" is for padding element
    # TODO: There is a bug in tf2.3: If you set mask_zero=True, GPU and CPU implementations return different values
    # TODO: It seems fixed in tf-nightly. See tf-bugs/gru-bug.py. Try it again in tf2.4
    items_branch = tf.keras.layers.Embedding(n_items + 1, Settings.EMBEDDING_DIM, mask_zero=False)(items_branch)

    # Process item inputs with a RNN layer
    items_branch = tf.keras.layers.GRU(64, return_sequences=True)(items_branch)
    #items_branch = tf.keras.layers.GRU(64, return_sequences=True, activation='relu')(items_branch)
    print(">>>>" , items_branch)

    # Flat to (batch size, -1) the RNN layer output
    items_branch = tf.keras.layers.Flatten()( items_branch )

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    # Conver to one-hot
    n_customers = len(customer_labels.labels)
    customer_branch = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, n_customers), name='one_hot_customer_encoding')(customer_input)

    # Concatenate the processed items sequence with the customer
    classification_branch = tf.keras.layers.Concatenate()( [ customer_branch , items_branch ] )

    # Process all input in Dense:
    classification_branch = tf.keras.layers.Dense(256)(classification_branch)

    # Do the classification
    classification_branch = tf.keras.layers.Dense(n_items, activation='softmax')(classification_branch)

    return tf.keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)
