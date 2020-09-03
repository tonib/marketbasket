import tensorflow as tf
from product_labels import ProductLabels

def create_model(product_labels: ProductLabels) -> tf.keras.Sequential:
    n_items = len(product_labels.labels)
    return tf.keras.Sequential([
        tf.keras.layers.Input(sparse=True, shape=n_items, name="input"),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(n_items, activation='softmax')
    ])
