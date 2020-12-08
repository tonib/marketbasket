from marketbasket.settings import settings
import tensorflow as tf
from .model_inputs import ModelInputs
from marketbasket.jsonutils import read_setting
from .conv_model import create_conv
from .rnn_model import create_rnn


def create_ensemble_model(inputs: ModelInputs) -> tf.keras.Model:

    # Get encoded sequence inputs
    encoded_inputs = inputs.get_all_as_sequence()

    # Apply RNN to inputs
    rnn_x = create_rnn(encoded_inputs)

    # Apply convolution to inputs
    conv_x = create_conv(encoded_inputs)

    # Merge convolution and RNN result
    x = tf.keras.layers.Concatenate()( [rnn_x , conv_x] )

    # "Ensemble" results
    ensemble_layer_size = read_setting( settings.model_config, 'ensemble_layer_size' , int , 512 )
    x = tf.keras.layers.Dense(ensemble_layer_size, activation='relu')(x)

    # Do the classification (logits)
    n_items = settings.features.items_sequence_feature().labels.length()
    x = tf.keras.layers.Dense(n_items, activation=None)(x)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)
