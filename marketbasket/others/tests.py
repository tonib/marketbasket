import tensorflow as tf

#@tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int64)])
def pad_sequence_left(sequences_batch: tf.RaggedTensor, width):
    """ Pad sequences with zeros on left side """

    sequences_batch = sequences_batch[:,-width:]  # Truncate rows to have at most `width` items
    pad_row_lengths = width - sequences_batch.row_lengths()
    pad_values = tf.zeros([(width * sequences_batch.nrows()) - tf.size(sequences_batch, tf.int64)], sequences_batch.dtype)
    padding = tf.RaggedTensor.from_row_lengths(pad_values, pad_row_lengths)
    return tf.concat([padding, sequences_batch], axis=1).to_tensor()

x = tf.ragged.constant( [ [ 1 , 2 ] , [] , [ 5 ,6 ,7 ] , [1] ] )
x = pad_sequence_left(x, 2)
print(x)

