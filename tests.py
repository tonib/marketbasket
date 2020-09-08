
import tensorflow as tf

tensor = tf.random.uniform((2,5))
print(tensor.numpy())

n_results = tf.constant(3)

sorted_indices = tf.argsort(tensor, direction='DESCENDING')
print("sorted", sorted_indices)

top_indices = sorted_indices[:,0:n_results]
print("top_indices", top_indices)

top_probabilities = tf.gather(tensor, top_indices, batch_dims=1)
print("top_probabilities", top_probabilities)

