
import tensorflow as tf

# tensor = tf.one_hot( [1, 2, 3] , 5 )
# print(tensor)

# sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 1], [1, 2]],
#                 values=[1, 2],
#                 dense_shape=[3, 4])
# print(sparse_tensor)

# sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 1], [2, 3]],
#                 dense_shape=[3, 4])
# print(sparse_tensor)

t = tf.ragged.constant([ [0, 1], [1, 2] , [1] ])
print(t)

t = tf.one_hot( t , 5 )
print(t)

t = tf.reduce_max( t , axis=1 )
print(t)
