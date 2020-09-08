import tensorflow as tf
from labels import Labels
from settings import Settings

@tf.function
def raged_lists_batch_to_multihot(ragged_lists_batch, multihot_dim):
    t = ragged_lists_batch.to_tensor(-1) # Default value = -1 -> one_hot will not assign any one
    t = tf.one_hot( t , multihot_dim )
    t = tf.reduce_max( t , axis=1 )
    return t

def create_model(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    n_items = len(item_labels.labels)
    n_customers = len(customer_labels.labels)

    # Input for input items will be a multihot array
    items_input = tf.keras.layers.Input(shape=(None, ), name='input_items_idx', dtype=tf.int64, ragged=True)
    encoded_items_layer = tf.keras.layers.Lambda(lambda x: raged_lists_batch_to_multihot(x, n_items), name="multi_hot_customer_encoding")
    encoded_items = encoded_items_layer(items_input)

    if Settings.N_MAX_CUSTOMERS > 0:

        # Customer index
        customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)

        # Conver to one-hot
        customer_encoder_layer = tf.keras.layers.Lambda(lambda x: tf.one_hot(x, n_customers), name='one_hot_customer_encoding')
        customer_input_encoded = customer_encoder_layer(customer_input)

        inputs = [ items_input , customer_input ]
        input_layer = tf.keras.layers.Concatenate(axis=1)( [ encoded_items , customer_input_encoded ] )
    else:
        inputs = [items_input]
        input_layer = encoded_items
    
    x = tf.keras.layers.Dense(256)(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(n_items, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
