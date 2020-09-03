
from model import create_model
import tensorflow as tf
from labels import Labels

# Load product labels
product_labels = Labels.load()

model = create_model(product_labels)

# Load the previously saved weights
latest_cp = tf.train.latest_checkpoint('model/checkpoints')
model.load_weights(latest_cp)

# Save full model
model.save('model/exported_model')

