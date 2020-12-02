import tensorflow as tf
from .model_inputs import ModelInputs
from marketbasket.settings import settings
from marketbasket.feature import Feature

@tf.function
def raged_lists_batch_to_multihot(ragged_lists_batch: tf.RaggedTensor, multihot_dim: int) -> tf.Tensor:
    """ Maps a batch of label indices to a batch of multi-hot ones """
    t = ragged_lists_batch.to_tensor(-1) # Default value = -1 -> one_hot will not assign any one
    t = tf.one_hot( t , multihot_dim )
    t = tf.reduce_max( t , axis=1 )
    return t

def input_to_one_hot_layer(input, feature: Feature) -> tf.keras.layers.Lambda:
    # START HERE: replace this lambda by a keras standard layer
    return tf.keras.layers.Lambda(lambda input: tf.one_hot(input, feature.labels.length()), name='one_hot_' + feature.name)

def create_dense_model(inputs: ModelInputs) -> tf.keras.Model:
    """ Create a DNN for classification """

    for input in inputs.inputs:
        print(input)
    # This model only supports a single sequence feature input (items labels). It will be a multihot array
    items_input = inputs.item_labels_input()
    items_feature = settings.features.items_sequence_feature()
    n_items = items_feature.labels.length()
    encoded_items_layer = tf.keras.layers.Lambda(lambda x: raged_lists_batch_to_multihot(x, n_items), name="multi_hot_" + 
        items_feature.name)
    encoded_items_output = encoded_items_layer(items_input)

    # Append transactions level features, encoded as one hot
    encoded_inputs = [ encoded_items_output ]
    for feature in settings.features.transaction_features():
        feature_input = inputs.by_feature[feature]
        encode_layer = input_to_one_hot_layer(feature_input, feature)
        encode_layer_output = encode_layer( inputs.by_feature[feature] )
        encoded_inputs.append( encode_layer_output )

    input_layer = tf.keras.layers.Concatenate(axis=1)( encoded_inputs )
    
    x = tf.keras.layers.Dense(1024, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Classification (logits)
    x = tf.keras.layers.Dense(n_items, activation=None)(x)

    return tf.keras.Model(inputs=inputs.inputs, outputs=x)