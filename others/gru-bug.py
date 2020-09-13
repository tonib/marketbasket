
import tensorflow as tf

# Issue with mask_zero=True:

def create_model():
    items_input = tf.keras.layers.Input(shape=[None], name='input_items_idx', dtype=tf.int64)
    embedding = tf.keras.layers.Embedding(10, 2, mask_zero=True)(items_input)
    gru = tf.keras.layers.LSTM(3, return_sequences=True)(embedding)

    return tf.keras.Model(inputs=[items_input], outputs=gru)

model_gpu = create_model()
input = tf.constant( [[1, 2, 3, 0 , 0 ]] )
print( "gpu result, original)", model_gpu( input ) )

with tf.device("CPU:0"):
    model_cpu = create_model()
    model_cpu.set_weights( model_gpu.get_weights() )
    print( "cpu result, from gpu weights", model_cpu( input ) )

    model_cpu = create_model()
    print( "cpu result, fresh new model", model_cpu( input ) )


# Output: GPU is masked, but CPU not
# [[[-0.00086556 -0.00042487 -0.00061207]
#   [ 0.00514142 -0.00360133 -0.002056  ]
#   [ 0.00602295 -0.00214107  0.00115865]
#   [ 0.          0.          0.        ]
#   [ 0.          0.          0.        ]]], shape=(1, 5, 3), dtype=float32)
# cpu result, from gpu weights tf.Tensor(
# [[[-0.00086556 -0.00042487 -0.00061207]
#   [ 0.00514142 -0.00360133 -0.002056  ]
#   [ 0.00602295 -0.00214107  0.00115865]
#   [ 0.00602295 -0.00214107  0.00115865]
#   [ 0.00602295 -0.00214107  0.00115865]]], shape=(1, 5, 3), dtype=float32)
# cpu result, fresh new model tf.Tensor(
# [[[ 0.00465276 -0.00134624  0.00086153]
#   [ 0.00147694 -0.00356608 -0.00022609]
#   [-0.00143589 -0.00348573 -0.00097939]
#   [-0.00143589 -0.00348573 -0.00097939]
#   [-0.00143589 -0.00348573 -0.00097939]]], shape=(1, 5, 3), dtype=float32)
