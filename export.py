from marketbasket.settings import settings # Setup. Must to be first
from marketbasket.model.model import create_model
import tensorflow as tf
from marketbasket.labels import Labels
from datetime import datetime

print(datetime.now(), "Process start: Export")

settings.features.load_label_files()
model = create_model()

# Load weights from last train checkpoint
latest_cp = tf.train.latest_checkpoint( settings.get_model_path('checkpoints') )
model.load_weights(latest_cp)

model.summary()

# Save full model
model.save( settings.get_model_path('exported_model'), include_optimizer=False )

print(datetime.now(), "Process end: Export")
