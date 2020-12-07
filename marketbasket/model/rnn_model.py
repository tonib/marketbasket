from marketbasket.settings import settings
import tensorflow as tf
from marketbasket.model.model_inputs import ModelInputs
from marketbasket.jsonutils import read_setting

def create_model_rnn(inputs: ModelInputs) -> tf.keras.Model:

    # Get encoded sequence inputs
    sequence_inputs = inputs.encode_inputs_set( settings.features.sequence_features() )

    # Get encoded transaction inputs, concatenated, and reapeat for each timestep
    transaction_inputs = inputs.encode_inputs_set( settings.features.transaction_features(), concatenate=True, 
        n_repeats=settings.sequence_length )
    
    # Concatenate sequence and transactions features on each timestep
    encoded_inputs = ModelInputs.merge_inputs_set( sequence_inputs + transaction_inputs )

    # Model settings
    layer_size = read_setting( settings.model_config, 'layer_size' , int , 128 )
    dropout_ratio = read_setting( settings.model_config, 'dropout_ratio' , float , 0.2 )

    # Define RNN
    rnn_layer = tf.keras.layers.GRU(layer_size)
    x = tf.keras.layers.Bidirectional(rnn_layer, name="rnn")(encoded_inputs)
    if dropout_ratio > 0:
        x = tf.keras.layers.Dropout(dropout_ratio, name="dropout")(x)

    # Do the classification (Logits)
    n_items = settings.features.items_sequence_feature().labels.length()
    x = tf.keras.layers.Dense(n_items, activation=None)(x)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)
