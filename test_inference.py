import tensorflow as tf
from marketbasket.predict import Prediction
from marketbasket.transaction import Transaction

#tf.config.run_functions_eagerly(True)

# with tf.device("CPU:0"):
predictor = Prediction()

# Test
# r = predictor.predict_single( Transaction.from_labels( ['4333' ], '[UNKNOWN]' ) , 10 )
# print(r)

# Test empty sequences in batch
batch = [ Transaction.from_labels( [ 'achilipu', 'arriquitaun' ], '[UNKNOWN]' ) , Transaction.from_labels( ['4333' ], '[UNKNOWN]' ),  
          Transaction.from_labels( [ 'arriquitaun' , '4333', 'achilipu' ], '[UNKNOWN]' ), ]
r = predictor.predict_batch( batch , 10 )
print(r)

# Test all sequences empty
# batch = [ Transaction.from_labels( ['achilipu', 'arriquitaun' ], '[UNKNOWN]' ) , Transaction.from_labels( ['notexists' ], '[UNKNOWN]' ) ]
# r = predictor.predict_batch( batch , 10 )
# print(r)
