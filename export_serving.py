from marketbasket.settings import settings # Setup, do not remove it
import tensorflow as tf
from marketbasket.predict import Prediction
from datetime import datetime

print(datetime.now(), "Process start: Export serving")

# See:
# https://www.tensorflow.org/api_docs/python/tf/saved_model/save
# https://www.tensorflow.org/guide/saved_model#saving_a_custom_model
# https://stackoverflow.com/questions/56659949/saving-a-tf2-keras-model-with-custom-signature-defs

p = Prediction()

#tf.saved_model.SaveTest.test_captures_unreachable_variable

#print(p._run_model_prediction.get_concrete_function().graph.as_graph_def())

# TODO: It seems the defautl signature name is "serving_default"...
tf.saved_model.save(p, settings.get_model_path('serving_model/1/'), signatures={ 'predict': p.run_model_single })

print(datetime.now(), "Process end: Export serving")
