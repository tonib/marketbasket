from marketbasket.settings import settings # Setup. Must to be first
from marketbasket.model import create_model
import tensorflow as tf
from marketbasket.labels import Labels
from datetime import datetime

print(datetime.now(), "Process start: Export")

# Load product labels
item_labels = Labels.load(Labels.item_labels_path())
customer_labels = Labels.load(Labels.customer_labels_path())

model = create_model(item_labels, customer_labels)

# Load weights from last train checkpoint
latest_cp = tf.train.latest_checkpoint( settings.get_model_path('checkpoints') )
model.load_weights(latest_cp)

model.summary()

# Save full model
model.save( settings.get_model_path('exported_model'), include_optimizer=False )

print(datetime.now(), "Process end: Export")
