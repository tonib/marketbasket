from marketbasket.settings import settings
import tensorflow as tf
from .model_inputs import ModelInputs
from marketbasket.jsonutils import read_setting

# Same as create_model_convolutional, with two stacked conv1d
def create_model_convolutional(inputs: ModelInputs) -> tf.keras.Model:

    # Get encoded sequence inputs
    encoded_inputs = inputs.get_all_as_sequence()

    # Not really a convolutional. It's a RNN + Convolutional ensemble

    # Model settings
    n_layers = read_setting( settings.model_config, 'n_layers' , int , 2 )
    layer_size = read_setting( settings.model_config, 'layer_size' , int , 128 )
    kernel_size = read_setting( settings.model_config, 'kernel_size' , int , 4 )
    strides = read_setting( settings.model_config, 'strides' , int , 1 )

    # Convolution
    x = encoded_inputs
    for i in range(n_layers):
        print(x, layer_size, kernel_size)
        x = tf.keras.layers.Conv1D(layer_size, kernel_size, strides=strides, activation='relu', name="conv_" + str(i))(x)

    # Flatten convolution outputs
    x = tf.keras.layers.Flatten()(x)

    # Do the classification (logits)
    n_items = settings.features.items_sequence_feature().labels.length()
    x = tf.keras.layers.Dense(n_items, activation=None)(x)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)
