from marketbasket.settings import settings # Setup. Must to be first
from marketbasket.labels import Labels
from datetime import datetime
from tensorflow.python.framework.ops import disable_eager_execution
from marketbasket.model.model import create_model, ModelType
import marketbasket.dataset as dataset
from real_eval import run_real_eval
from marketbasket.predict import Prediction
from focal_loss import SparseCategoricalFocalLoss
from marketbasket.class_weights import ClassWeights
from datetime import datetime
import argparse
import tensorflow as tf

# To test with GPU disabled set environment variable CUDA_VISIBLE_DEVICES=-1

print(datetime.now(), "Process start: Rating model train")
settings.features.load_label_files()
settings.print_summary()

# Define train dataset
train_dataset = dataset.get_dataset(True, True)

# Define evaluation dataset
eval_dataset = dataset.get_dataset(True, False)

# We need the batches number in evaluation dataset, so here is:
# (This will be executed in eager mode)
n_eval_batches = dataset.n_batches_in_dataset(eval_dataset)

# Create model
model = create_model(True)

model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=SparseCategoricalFocalLoss(gamma=3, from_logits=True),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=settings.get_model_path('logs'))

# Save checkpoints
checkpoint_file_format = settings.get_model_path() + '/rating_model_checkpoints/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_format,
                                                 save_weights_only=True,
                                                 verbose=1)

# Do real evaluation callback:
# TODO: Performance of this could be improved A LOT
predictor = Prediction(model)
class RealEvaluationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        run_real_eval(predictor)

# TF 2.3: Requires validation_steps. It seems a bug, as documentation says it can be None for TF datasets, but
# with None it throws exception

model.fit(train_dataset, 
        epochs=settings.n_epochs,
        callbacks=[tensorboard_callback, cp_callback, RealEvaluationCallback()], 
        validation_data=eval_dataset,
        validation_steps=n_eval_batches,
        verbose=settings.train_log_level)

print(datetime.now(), "Process end: Train")
