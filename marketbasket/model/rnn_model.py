from marketbasket.settings import settings
import tensorflow as tf
from marketbasket.model.model_inputs import ModelInputs
from marketbasket.jsonutils import read_setting
from marketbasket.model.dense_model import create_output_layer

def create_rnn(encoded_inputs):
    # Model settings
    layer_size = read_setting( settings.model_config, 'rnn_layer_size' , int , 128 )
    dropout_ratio = read_setting( settings.model_config, 'rnn_dropout_ratio' , float , 0.2 )
    bidirectional = read_setting( settings.model_config, 'rnn_bidirectional' , bool , True )
    rnn_n_layers = read_setting( settings.model_config, 'rnn_n_layers' , int , 1 )

    # Define layers
    x = encoded_inputs
    for i in range(rnn_n_layers):
        return_sequences = ( i < (rnn_n_layers - 1) )
        rnn_layer = tf.keras.layers.GRU(layer_size, name="rnn_" + str(i), return_sequences=return_sequences)
        if bidirectional:
            rnn_layer = tf.keras.layers.Bidirectional(rnn_layer, name="rnn_bidir")
        x = rnn_layer(x)

        if dropout_ratio > 0:
            x = tf.keras.layers.Dropout(dropout_ratio, name="dropout_" + str(i))(x)

    return x

def create_model_rnn(inputs: ModelInputs, rating_model: bool) -> tf.keras.Model:

    # Get encoded sequence inputs
    encoded_inputs = inputs.get_all_as_sequence()

    # Apply model
    x = create_rnn(encoded_inputs)

    # Output layer
    x = create_output_layer(inputs, x, rating_model)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)
