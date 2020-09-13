import tensorflow as tf

t = tf.constant([ "a", "b"])
t = tf.expand_dims(t, axis=0)
print(t)
print( tf.RaggedTensor.from_tensor(t) )
