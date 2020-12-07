import tensorflow as tf
from .model_inputs import ModelInputs
from marketbasket.settings import settings
from marketbasket.feature import Feature
from marketbasket.jsonutils import read_setting

"""
    DNN model definition
"""


def create_dense_model(inputs: ModelInputs) -> tf.keras.Model:
    """ Create a DNN for classification """

    # This model only supports a single sequence feature input (items labels). It will be a multihot array
    items_input = inputs.item_labels_input()
    items_feature = settings.features.items_sequence_feature()
    encoded_items_output = items_feature.encode_input(items_input, as_multihot=True)

    # Get transactions level features
    encoded_trn_inputs = inputs.encode_inputs_set( settings.features.transaction_features() )

    # Merge item labels and transaction features
    encoded_inputs = [ encoded_items_output ] + encoded_trn_inputs
    encoded_inputs = ModelInputs.merge_inputs_set( encoded_inputs )
    
    # Model settings
    n_layers = read_setting( settings.model_config, 'n_layers' , int , 2 )
    layer_size = read_setting( settings.model_config, 'layer_size' , int , 128 )
    activation = read_setting( settings.model_config, 'activation' , str , 'relu' )
    
    # Define DNN
    x = encoded_inputs
    for i in range(n_layers):
        x = tf.keras.layers.Dense(layer_size, name="dnn_" + str(i), activation=activation)(x)
    
    # Classification (logits)
    x = tf.keras.layers.Dense(items_feature.labels.length(), name="classification", activation=None)(x)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)