
from model import create_model
import tensorflow as tf
from labels import Labels

# Load product labels
item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)

model = create_model(item_labels, customer_labels)

# Load weights from last train checkpoint
latest_cp = tf.train.latest_checkpoint('model/checkpoints')
model.load_weights(latest_cp)

# Save full model
model.save('model/exported_model')
