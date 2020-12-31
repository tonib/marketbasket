from marketbasket.settings import settings
import tensorflow as tf
from .model_inputs import ModelInputs
from marketbasket.jsonutils import read_setting
from .conv_model import create_conv
from .rnn_model import create_rnn
import marketbasket.model.dense_model as dense_model

def create_ensemble_model(inputs: ModelInputs, rating_model: bool) -> tf.keras.Model:

    # Get encoded sequence inputs
    encoded_inputs = inputs.get_all_as_sequence()

    # Apply RNN to inputs
    rnn_x = create_rnn(encoded_inputs)

    # Apply convolution to inputs
    conv_x = create_conv(encoded_inputs)

    # Apply dense
    # items_as_multihot = dense_model.items_as_multihot(inputs)
    # dense_x = dense_model.create_dense(items_as_multihot)

    # Merge convolution and RNN result
    #x = tf.keras.layers.Concatenate()( [rnn_x , conv_x, dense_x] )
    x = tf.keras.layers.Concatenate()( [rnn_x , conv_x] )

    # "Ensemble" results
    ensemble_layer_size = read_setting( settings.model_config, 'ensemble_layer_size' , int , 512 )
    if ensemble_layer_size > 0:
        x = tf.keras.layers.Dense(ensemble_layer_size, activation='relu')(x)

    # Output layer
    x = dense_model.create_output_layer(inputs, x, rating_model)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)
