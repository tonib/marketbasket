from marketbasket.settings import settings # Setup. Must to be first
from marketbasket.model.model import create_model
from marketbasket.labels import Labels
from datetime import datetime
from marketbasket.predict import Prediction
import tensorflow as tf

print(datetime.now(), "Process start: Export")

settings.features.load_label_files()
model = create_model()

# Load weights from last train checkpoint
latest_cp = tf.train.latest_checkpoint( settings.get_model_path('checkpoints') )
model.load_weights(latest_cp)

model.summary()

# Save full model
model.save( settings.get_model_path( Prediction.CANDIDATES_EXPORTED_MODEL_DIR ), include_optimizer=False )

print(datetime.now(), "Process end: Export")
