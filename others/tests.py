import tensorflow as tf

input = tf.ragged.constant( [ [1] , [2, 3] ] )
output = tf.constant([ # batch
    [ # batch element 0
        [ #timestep 0 probs
            0 , 1 , 3
        ],
        [ #timestep 1 probs
            2 , 3 , 4
        ]
    ],
    [ [ 5 , 6 , 7 ] , [ 8 , 9 , 10 ] ],  # Batch element 1
])

indices = input.row_lengths()
indices -= 1
#indices = tf.expand_dims(input_lengths, 1)
print("indices", indices)

probs = tf.gather(output, indices, batch_dims=1)
print("probs", probs)
