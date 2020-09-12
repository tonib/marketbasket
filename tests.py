
import tensorflow as tf
from labels import Labels
import model
from labels import Labels
from predict import Prediction
from model import pad_sequence

def create_model():
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64)
    embedding = tf.keras.layers.Embedding(10, 2, mask_zero=True)(items_input)
    gru = tf.keras.layers.LSTM(3, return_sequences=True)(embedding)

    return tf.keras.Model(inputs=[items_input], outputs=gru)

# See https://github.com/tensorflow/tensorflow/issues/30745

model_gpu = create_model()
input = tf.constant( [[1, 2, 3, 0 , 0 ]] )
print( "gpu result, original)", model_gpu( input ) )

with tf.device("CPU:0"):
    model_cpu = create_model()
    model_cpu.set_weights( model_gpu.get_weights() )
    print( "cpu result, from gpu weights", model_cpu( input ) )

    model_cpu = create_model()
    print( "cpu result, fresh new model", model_cpu( input ) )


