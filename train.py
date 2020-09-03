import tensorflow as tf
from product_labels import ProductLabels
from datetime import datetime
from tensorflow.python.framework.ops import disable_eager_execution
from settings import Settings

# We need the batches number in evaluation dataset, so here is:
# (This will be executed in eager mode)
train_dataset = tf.data.TFRecordDataset( [ 'data/dataset_eval.tfrecord' ] )
train_dataset = train_dataset.batch( Settings.BATCH_SIZE )
for n_eval_batches, _ in enumerate(train_dataset):
    pass
# print(f'Number of elements: {n_eval_batches}')
# exit()

# Be sure we use Graph mode (performance)
disable_eager_execution()

# Load product labels
product_labels = ProductLabels.load()
N_ITEMS = len( product_labels.labels )

# https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

# define your tfrecord again. Remember that you saved your image as a string.
keys_to_features = {
    'input': tf.io.SparseFeature('sparse_indices', 'sparse_values', tf.float32,  N_ITEMS, False),
    'output_label': tf.io.FixedLenFeature([], tf.int64)
}

@tf.function
def example_parse_function(proto_batch):
    # Load one example
    parsed_features = tf.io.parse_example(proto_batch, keys_to_features)
    return parsed_features['input'], parsed_features['output_label']

# Define train dataset
train_dataset = tf.data.TFRecordDataset( [ 'data/dataset_train.tfrecord' ] )
train_dataset = train_dataset.prefetch(10000)
train_dataset = train_dataset.shuffle(10000).batch( Settings.BATCH_SIZE )
train_dataset = train_dataset.map( example_parse_function , num_parallel_calls=8 )

# Define evaluation dataset
eval_dataset = tf.data.TFRecordDataset( [ 'data/dataset_eval.tfrecord' ] )
eval_dataset = eval_dataset.prefetch(10000)
eval_dataset = eval_dataset.batch( Settings.BATCH_SIZE )
eval_dataset = eval_dataset.map( example_parse_function )

model = tf.keras.Sequential([
    tf.keras.layers.Input(sparse=True, shape=N_ITEMS),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense( len(product_labels.labels) )
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Tensorboard
#logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Save checkpoints
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/checkpoint',
                                                 save_weights_only=True,
                                                 verbose=1)

# TF 2.3: Requires validation_steps. It seems a bug, as documentation says it can be None for TF datasets, but
# with None it throws exception
model.fit(train_dataset, 
          epochs=Settings.N_EPOCHS,
          callbacks=[tensorboard_callback, cp_callback], 
          validation_data=eval_dataset,
          validation_steps=n_eval_batches)


# for epoch in range(Settings.N_EPOCHS):
#     model.fit(train_dataset, 
#             callbacks=[tensorboard_callback, cp_callback])

# model.fit(train_dataset, 
#         callbacks=[tensorboard_callback, cp_callback],
#         epochs=Settings.N_EPOCHS)
