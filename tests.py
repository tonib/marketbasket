
import tensorflow as tf

#@tf.function
def test2( sequences_batch: tf.RaggedTensor, sequence_length: int ) -> tf.Tensor:
    print(sequences_batch)

    # Avoid sequences larger than sequence_length: Get last sequence_length of each sequence
    sequences_batch = sequences_batch[:,-sequence_length:]
    # Add one to indices, to reserve 0 index for padding
    sequences_batch = sequences_batch + 1
    # Convert to dense, padding zeros to the right
    sequences_batch = sequences_batch.to_tensor(0, shape=[None, sequence_length])
    return sequences_batch

#@tf.function
def test( sequences_batch: tf.RaggedTensor, sequence_length: int ) -> tf.Tensor:
    print(sequences_batch)

    # Avoid sequences larger than sequence_length: Get last sequence_length of each sequence
    sequences_batch = sequences_batch[:,-sequence_length:]
    print( sequences_batch )
    
    # Number of elements to pad on each batch row
    pad_row_lengths = sequence_length - sequences_batch.row_lengths()
    print( pad_row_lengths )

    # 1d with all elements to pad (pad element is temporally -1)
    n_pad_elements = (sequence_length * sequences_batch.nrows()) - tf.size(sequences_batch, tf.int64 )
    pad_values = tf.repeat(-1, n_pad_elements )
    print("pad_values", pad_values)

    # 2d with elements to padd on each batch row
    padding = tf.RaggedTensor.from_row_lengths(pad_values, pad_row_lengths)
    print("padding", padding)

    # Append the padding to the batch and return a dense tensor (pad right for CuDNNGRU/GPU compatiblity)
    sequences_batch = tf.concat([sequences_batch, padding], axis=1).to_tensor()

    # +1 to all elements. This will turn padding element zero and increase all other indices +1
    # Needed for keras embedding (padding element MUST to be the zero)
    return sequences_batch + 1

    # print(t)
    # print(t.nested_row_lengths())
    # print(t.bounding_shape())

    # t = t.to_tensor(-1)
    # print(t)

    # n_cols_to_pad = dim - tf.shape(t)[1] 
    # if n_cols_to_pad > 0:
    #     t = tf.pad( t , [ [0,0] , [n_cols_to_pad, 0] ], constant_values=-1 )


t = tf.ragged.constant( [ [1], [1, 3, 0], [2, 1], [] , [0, 1, 2, 3, 4, 5, 6, 7] ] )
# for _ in range(1000):
#     test(t, 3)
#     test(t, 10).numpy()

print( test2(t, 3).numpy() )
print( test2(t, 10).numpy() )

# t = tf.ragged.constant( [ [1], [1, 3, 0], [2, 1] ] )
# print( test(t, 4).numpy() )

