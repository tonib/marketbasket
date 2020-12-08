from marketbasket.settings import settings
import tensorflow as tf
from .model_inputs import ModelInputs
from marketbasket.jsonutils import read_setting

def create_conv(encoded_inputs):
    # Model settings
    n_layers = read_setting( settings.model_config, 'conv_n_layers' , int , 2 )
    layer_size = read_setting( settings.model_config, 'conv_layer_size' , int , 128 )
    kernel_size = read_setting( settings.model_config, 'conv_kernel_size' , int , 4 )
    strides = read_setting( settings.model_config, 'conv_strides' , int , 1 )

    # Convolution
    x = encoded_inputs
    for i in range(n_layers):
        print(x, layer_size, kernel_size)
        x = tf.keras.layers.Conv1D(layer_size, kernel_size, strides=strides, activation='relu', name="conv_" + str(i))(x)

    # Flatten convolution outputs
    x = tf.keras.layers.Flatten()(x)

    return x

# Same as create_model_convolutional, with two stacked conv1d
def create_model_convolutional(inputs: ModelInputs) -> tf.keras.Model:

    # Get encoded sequence inputs
    encoded_inputs = inputs.get_all_as_sequence()

    # Apply model
    x = create_conv(encoded_inputs)

    # Do the classification (logits)
    n_items = settings.features.items_sequence_feature().labels.length()
    x = tf.keras.layers.Dense(n_items, activation=None)(x)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)
