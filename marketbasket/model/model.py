from marketbasket.settings import settings, ModelType
import tensorflow as tf
from marketbasket.labels import Labels
from .gpt import *
from .model_inputs import ModelInputs
from marketbasket.model.dense_model import create_dense_model
from marketbasket.model.rnn_model import create_model_rnn
from marketbasket.model.conv_model import create_model_convolutional
from marketbasket.model.ensemble_model import create_ensemble_model

def create_model(rating_model: bool) -> tf.keras.Model:
    """ Returns the model """
    inputs = ModelInputs()

    if settings.model_type == ModelType.DENSE:
        return create_dense_model(inputs, rating_model)
    elif settings.model_type == ModelType.RNN:
        return create_model_rnn(inputs, rating_model)
    elif settings.model_type == ModelType.CONVOLUTIONAL:
        return create_model_convolutional(inputs, rating_model)
    elif settings.model_type == ModelType.ENSEMBLE:
        return create_ensemble_model(inputs, rating_model)
    else:
        raise Exception("Unknown model type " + settings.model_type)

    # if settings.model_type == ModelType.RNN:
    #     return create_model_rnn(item_labels, customer_labels)
    # elif settings.model_type == ModelType.DENSE:
    #     return create_dense_model(inputs)
    # elif settings.model_type == ModelType.CONVOLUTIONAL:
    #     # Pending
    #     return create_model_convolutional_v2(item_labels, customer_labels)
    # elif settings.model_type == ModelType.GPT:
    #     return create_model_gpt_raw(item_labels, customer_labels)
    # else:
    #     raise Exception("Unknown model type" + settings.model_type)

##########################################################################################
# GPT
##########################################################################################

def create_model_gpt_with_conv(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.customers_embedding_dim)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.sequence_length)(customer_branch)

    # Input for input items will be a sequence of embeded items
    n_items = len(item_labels.labels)
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)

    # Items embedding
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.items_embedding_dim)(items_branch)

    # Append positional encoding for each item sequence index
    transformer_branch = AddPositionEmbedding(settings.sequence_length, settings.items_embedding_dim)(items_branch)
    # Concatenate embedded customer and items on each timestep
    transformer_branch = tf.keras.layers.Concatenate()( [ transformer_branch , customer_branch ] )
    # Transformer decoder
    num_heads = 8  # Number of attention heads
    feed_forward_dim = 128  # Hidden layer size in feed forward network inside transformer
    transformer_branch = TransformerBlock(settings.items_embedding_dim + settings.customers_embedding_dim, num_heads, feed_forward_dim)(transformer_branch)

    # Add customer to input (without position encoding)
    convolution_branch = tf.keras.layers.Concatenate()( [ items_branch , customer_branch ] )
    #print(">>>> convolution_branch", convolution_branch)
    # Convolution
    convolution_branch = tf.keras.layers.Conv1D(64, 4, activation='relu')(convolution_branch)
    # Flatten convolution outputs
    convolution_branch = tf.keras.layers.Flatten()(convolution_branch)
    # Repeat for each timestep
    convolution_branch = tf.keras.layers.RepeatVector(settings.sequence_length)(convolution_branch)

    # Concatenate transformer output and convolution output on each timestep
    classification_branch = tf.keras.layers.Concatenate()( [ transformer_branch , convolution_branch ] )

    # Process transformer output and context 
    classification_branch = tf.keras.layers.Dense(512, activation='relu')(classification_branch)

    # Classification (logits)
    classification_branch = layers.Dense(n_items)(classification_branch)

    return keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)


def create_model_gpt_raw(item_labels: Labels, customer_labels: Labels) -> tf.keras.Model:

    # Customer index
    customer_input = tf.keras.layers.Input(shape=(), name='customer_idx', dtype=tf.int64)
    n_customers = len(customer_labels.labels)
    # Embed customer
    customer_branch = tf.keras.layers.Embedding(n_customers, settings.customers_embedding_dim)(customer_input)
    # Repeat embedded customer for each timestep
    customer_branch = tf.keras.layers.RepeatVector(settings.sequence_length)(customer_branch)

    # Input for input items will be a sequence of embedded items
    n_items = len(item_labels.labels)
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64, ragged=True)
    # Pad items sequence:
    items_branch = tf.keras.layers.Lambda(lambda x: pad_sequence_right(x), name="padded_sequence")(items_input)

    # Items embedding
    # +1 in "n_items + 1" is for padding element. Value zero is reserved for padding
    items_branch = tf.keras.layers.Embedding(n_items + 1, settings.items_embedding_dim)(items_branch)

    # Append positional encoding for each item sequence index
    transformer_branch = AddPositionEmbedding(settings.sequence_length, settings.items_embedding_dim)(items_branch)
    # Concatenate embedded customer and items on each timestep
    transformer_branch = tf.keras.layers.Concatenate()( [ transformer_branch , customer_branch ] )
    # Transformer decoder
    num_heads = 8  # Number of attention heads
    feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer
    transformer_branch = TransformerBlock(settings.items_embedding_dim + settings.customers_embedding_dim, num_heads, feed_forward_dim)(transformer_branch)

    # Classification (logits)
    classification_branch = layers.Dense(n_items)(transformer_branch)

    return keras.Model(inputs=[items_input, customer_input], outputs=classification_branch)
