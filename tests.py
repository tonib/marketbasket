
import tensorflow as tf

tensor = tf.random.uniform((2,5))
print(tensor.numpy())

indices = tf.ragged.constant([ [0] , [1, 2] ], dtype=tf.int64).to_tensor(-1)
print(indices.numpy())

row_indices = tf.range(0, tf.shape( indices )[0] , dtype=tf.int64)
print( row_indices)
# row_indices = tf.transpose( row_indices )
# print( row_indices )
row_indices = tf.repeat( row_indices, tf.shape(indices)[1] )
print( row_indices.numpy() )

indices = tf.reshape( indices , shape=[-1] )
print( "reshaped row indices", indices )

indices = tf.stack( [row_indices, indices], axis = 1 )
print( "stacked indices", indices )

# Remove indices out of bound (tensor_scatter_nd_update will fail in CPU with these indices)
#print( "condition", indices[:,1] != -1 )
gather_idxs = tf.where( indices[:,1] != -1 )
#gather_idxs = tf.reshape( gather_idxs , (-1) )
print( "gather_idxs", gather_idxs.numpy() )

# Removed -1 indices
indices = tf.gather_nd(indices, gather_idxs)
print( "indices without -1", indices )

#print( "*" , tf.where(indices != -1).numpy() )

# updates = tf.ones( [indices.shape[0]] , dtype=tf.float32 )
# updates = -updates
updates = tf.repeat(-1.0, tf.shape(indices)[0] )
print(updates.numpy())

new_t = tf.tensor_scatter_nd_update( tensor, indices, updates)
print(new_t.numpy())


# t = tf.ragged.constant( [ [1, 2] , [2] , [], [0,1, 2] ] )
# t = t.to_tensor(-1)
# print(t.numpy())
# t = tf.one_hot( t , 4 )
# print(t.numpy())
# t = tf.reduce_max( t , axis=1 )
# print(t.numpy())