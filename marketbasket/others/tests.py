
import tensorflow as tf

@tf.function
def repeat_batch_inputs(input_batch, n_items_result: int):
    # Repeat transaction inputs for each predicted top item 
    result = []
    for input in input_batch:
        if isinstance(input, tf.RaggedTensor):
            # tf.repeat cannot be used with ragged tensors...            
            # ex: input = [ [1] , [2,3] ] , n_items_result = 2
            row_lengths = input.row_lengths() # -> [ 1, 2 ]
            row_lengths = tf.repeat( row_lengths , [n_items_result], axis=0 ) # -> [ 1 , 1 , 2 , 2 ]
            input = tf.tile( input, [1, n_items_result] ) #  -> [ [1, 1] , [2, 3, 2, 3] ]
            input = input.flat_values # -> [ 1, 1 , 2, 3, 2, 3 ]
            input = tf.RaggedTensor.from_row_lengths(input, row_lengths) # -> [ [1], [1], [2, 3], [2, 3] ]
        else:
            input = tf.repeat(input, [n_items_result], axis=0 )
        result.append(input)
    return result

x = [
    tf.constant( [ 1 , 2 , 3 ] ),
    tf.constant( [ 4 , 5 , 6 ] ),
    tf.ragged.constant( [ [1] , [2, 3], [4] ] )
]
print( repeat_batch_inputs(x, 3) )
