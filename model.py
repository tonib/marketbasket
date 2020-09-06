import tensorflow as tf
from labels import Labels
from settings import Settings

def create_model(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    n_items = len(item_labels.labels)
    n_customers = len(customer_labels.labels)

    # Input for input items will be a multihot array
    items_input = tf.keras.layers.Input(shape=n_items, name='input_items_idx')

    if Settings.N_MAX_CUSTOMERS > 0:
        # Customer index
        customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
        # Conver to one-hot
        customer_input_encoded = tf.one_hot(customer_input, n_customers, name='onehot_customer_idx')

        inputs = [ items_input , customer_input ]
        input_layer = tf.keras.layers.Concatenate(axis=1)( [ items_input , customer_input_encoded ] )
    else:
        inputs = [items_input]
        input_layer = items_input
    
    x = tf.keras.layers.Dense(256)(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(n_items, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
