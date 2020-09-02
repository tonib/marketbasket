import tensorflow as tf
from product_labels import ProductLabels
from datetime import datetime
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

N_MAX_ITEMS = 100

# https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

# define your tfrecord again. Remember that you saved your image as a string.
keys_to_features = {
    'input': tf.io.SparseFeature('sparse_indices', 'sparse_values', tf.float32,  100, False),
    'output_label': tf.io.FixedLenFeature([], tf.int64)
}

@tf.function
def train_parse_function(proto_batch):
    # Load one example
    parsed_features = tf.io.parse_example(proto_batch, keys_to_features)
    return parsed_features['input'], parsed_features['output_label']

# Define dataset
dataset = tf.data.TFRecordDataset( [ 'data/dataset.tfrecord' ] )
dataset = dataset.prefetch(10000)
dataset = dataset.shuffle(10000).batch(128)
dataset = dataset.map( train_parse_function , num_parallel_calls=8 )

product_labels = ProductLabels.load()
print( len(product_labels.labels) )

model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(N_MAX_ITEMS)),
    tf.keras.layers.Input(sparse=True, shape=100),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense( len(product_labels.labels) )
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit( dataset, callbacks=[tensorboard_callback] , epochs=30)
