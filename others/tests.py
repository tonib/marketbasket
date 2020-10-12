import tensorflow as tf

n_sequences = 4
n_results = 10
label_predictions = tf.zeros( [n_sequences, n_results] , dtype=tf.string )
print(label_predictions)

probs_predictions = tf.zeros( [n_sequences, n_results] , dtype=tf.float32 )
print(probs_predictions)
