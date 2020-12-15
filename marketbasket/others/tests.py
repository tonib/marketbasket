
import tensorflow as tf

# a = tf.keras.Input(shape=(), name='a') # 'a' becomes a key in the dict.
# b = tf.keras.Input(shape=(), name='b')
a = tf.keras.Input(shape=())
b = tf.keras.Input(shape=())
model = tf.keras.Model([a, b], [a])

@tf.function
def run_model(inputs):
    return model(inputs) + 10


# input = { 'a': tf.constant( [1] ), 'b': tf.constant([2]) } 
# input_reversed = { 'b': tf.constant( [2] ), 'a': tf.constant([1]) } 

input = {}
input['a'] = tf.constant( [1] )
input['b'] = tf.constant( [2] )
input_reversed = {}
input_reversed['b'] = tf.constant( [2] )
input_reversed['a'] = tf.constant( [1] )

print( model( input ) )
print( model( input_reversed ) )

print( run_model( input ) )
print( run_model( input_reversed ) )
