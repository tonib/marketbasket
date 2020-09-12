import tensorflow as tf
from predict import Prediction
from transaction import Transaction

#tf.config.run_functions_eagerly(True)

# with tf.device("CPU:0"):
predictor = Prediction()

# Test
r = predictor.predict_single( Transaction.from_labels( ['4333' ], '[UNKNOWN]' ) , 10 )
print(r)

# Test
# #batch = [ Transaction.from_labels( ['achilipu', 'arriquitaun' ], '[UNKNOWN]' ) , Transaction.from_labels( ['4333' ], '[UNKNOWN]' ) ]
# batch = [ Transaction.from_labels( ['4333', '21730' ], '[UNKNOWN]' ) , Transaction.from_labels( ['4333' ], '[UNKNOWN]' ) ]
# r = predictor.predict_batch( batch , 10 )
# print(r)
