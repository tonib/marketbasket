import tensorflow as tf

seq_len = 2

batch = tf.constant( [ [[ 10 , 11 ] , [ 12 , 13 ]] , [[ 14 , 15 ] , [ 16 , 17 ]] ] )

context_features = tf.constant( [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] ] )
context_features = tf.expand_dims(context_features, 1 )
print( "context_features", context_features )

repeats = tf.repeat(context_features, seq_len, axis=1 )
print("repeats", repeats)

result = tf.concat( [batch, repeats] , 2 )
print("result", result)
