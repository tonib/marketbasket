import tensorflow as tf
from labels import Labels
from settings import Settings

def create_model(product_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    n_items = len(product_labels.labels)
    n_customers = len(customer_labels.labels)

    # Input for input items will be a multihot array
    items_input = tf.keras.layers.Input(shape=n_items, name='input_items_idx')

    if Settings.N_MAX_CUSTOMERS > 0:
        # Input for customer will be a one-hot array
        customer_input = tf.keras.layers.Input(shape=n_customers, name='customer_idx')
        inputs = [ items_input , customer_input ]
        input_layer = tf.keras.layers.Concatenate(axis=1)( inputs )
    else:
        inputs = [items_input]
        input_layer = items_input
    
    x = tf.keras.layers.Dense(256)(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(n_items, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
