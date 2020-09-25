import tensorflow as tf

# batch size = 2, seq size = 3 embeding size = 4
embedded_items = tf.constant( [ [[1,2,3,4] , [5,6,7,8] , [9,10,11,12]] , [[1,2,3,44] , [5,6,7,88] , [9,10,11,122]] ] )
# batch size = 2, embeding size = 2
embedded_customer = tf.constant( [ [30, 31] , [40, 41] ] )

embedded_items_shape = tf.shape(embedded_items)
batch_size = embedded_items_shape[0]
sequence_length = embedded_items_shape[1]

embedded_customer = tf.repeat(embedded_customer, sequence_length , axis=0 )
embedded_customer = tf.reshape( embedded_customer , [ batch_size , sequence_length , -1 ] )

#print( tf.shape(t1) )
print(embedded_customer)
result = tf.concat( [embedded_items, embedded_customer] , axis=2)
print(result)
