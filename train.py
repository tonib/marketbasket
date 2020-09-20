import tensorflow as tf
from labels import Labels
from datetime import datetime
from tensorflow.python.framework.ops import disable_eager_execution
from settings import Settings
from model import create_model
from dataset import DataSet
from real_eval import run_real_eval
from predict import Prediction
from focal_loss import SparseCategoricalFocalLoss

# To test with GPU disabled set environment variable CUDA_VISIBLE_DEVICES=-1

# We need the batches number in evaluation dataset, so here is:
# (This will be executed in eager mode)
n_eval_batches = DataSet.n_eval_batches()

# Load labels
item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)

# Setup data set global variables
DataSet.setup_feature_keys(item_labels, customer_labels)

# Define train dataset
train_dataset = DataSet.load_train_dataset()

# Define evaluation dataset
eval_dataset = DataSet.load_eval_dataset()

# Create model
model = create_model(item_labels, customer_labels)


model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              loss=SparseCategoricalFocalLoss(gamma=2, from_logits=False),
              metrics=['accuracy'])

model.summary()

# Tensorboard
#logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "model/logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Save checkpoints
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model/checkpoints/cp-{epoch:04d}.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)

# Do real evaluation callback:
predictor = Prediction(model)
class RealEvaluationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        run_real_eval(predictor)

# TF 2.3: Requires validation_steps. It seems a bug, as documentation says it can be None for TF datasets, but
# with None it throws exception

model.fit(train_dataset, 
        epochs=Settings.N_EPOCHS,
        callbacks=[tensorboard_callback, cp_callback, RealEvaluationCallback()], 
        validation_data=eval_dataset,
        validation_steps=n_eval_batches)
