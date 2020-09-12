import tensorflow as tf
from predict import Prediction

# See:
# https://www.tensorflow.org/api_docs/python/tf/saved_model/save
# https://www.tensorflow.org/guide/saved_model#saving_a_custom_model
# https://stackoverflow.com/questions/56659949/saving-a-tf2-keras-model-with-custom-signature-defs

p = Prediction()

#tf.saved_model.SaveTest.test_captures_unreachable_variable

#print(p._run_model_prediction.get_concrete_function().graph.as_graph_def())

tf.saved_model.save(p, 'model/serving_model', signatures={ "predicct": p._top_predictions_tensor })
