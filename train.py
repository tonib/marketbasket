import tensorflow as tf
from labels import Labels
from datetime import datetime
from tensorflow.python.framework.ops import disable_eager_execution
from settings import Settings
from model import create_model

# We need the batches number in evaluation dataset, so here is:
# (This will be executed in eager mode)
train_dataset = tf.data.TFRecordDataset( [ Settings.EVAL_DATASET_FILE ] )
train_dataset = train_dataset.batch( Settings.BATCH_SIZE )
for n_eval_batches, _ in enumerate(train_dataset):
    pass
# print(f'Number of elements: {n_eval_batches}')
# exit()

# Be sure we use Graph mode (performance)
disable_eager_execution()

# Load labels
item_labels = Labels.load(Settings.ITEM_LABELS_FILE)
N_ITEMS = len( item_labels.labels )
customer_labels = Labels.load(Settings.CUSTOMER_LABELS_FILE)
N_CUSTOMERS = len( customer_labels.labels )

# Feature mappings for training
keys_to_features = {
    'input_items_idx': tf.io.RaggedFeature(tf.int64, 'input_items_idx'),
    'output_item_idx': tf.io.FixedLenFeature([], tf.int64)
}
if Settings.N_MAX_CUSTOMERS > 0:
    keys_to_features['customer_idx'] = tf.io.FixedLenFeature([], tf.int64)


@tf.function
def raged_lists_batch_to_multihot(ragged_lists_batch, multihot_dim):
    t = tf.one_hot( ragged_lists_batch , multihot_dim )
    t = tf.reduce_max( t , axis=1 )
    return t

@tf.function
def example_parse_function(proto_batch):
    # Load one example
    parsed_features = tf.io.parse_example(proto_batch, keys_to_features)

    # Map batch of list of item indices to a batch of multi-hot arrays
    # Ex [ [1, 2] , [2, 3] ] -> [ [ 0 , 1 , 1 , 0 ] , [ 0 , 0 , 1 , 1 ] ]
    items_input = parsed_features['input_items_idx']
    items_input = raged_lists_batch_to_multihot( items_input , N_ITEMS )

    if Settings.N_MAX_CUSTOMERS > 0:
        # Batch of customer indices. Ex: [1, 2]
        customer = parsed_features['customer_idx']
        # Batch of one hot encodings. Ex: [ [0 , 1 , 0], [0, 0, 1] ]
        customer = tf.one_hot(customer, N_CUSTOMERS)
        input = { 'input_items_idx': items_input , 'customer_idx': customer }
    else:
        input = items_input

    # Return tuple (net input, expected output)
    return input, parsed_features['output_item_idx']

# Define train dataset
train_dataset = tf.data.TFRecordDataset( [ Settings.TRAIN_DATASET_FILE ] )
train_dataset = train_dataset.prefetch(10000)
train_dataset = train_dataset.shuffle(10000).batch( Settings.BATCH_SIZE )
train_dataset = train_dataset.map( example_parse_function , num_parallel_calls=8 )

# Define evaluation dataset
eval_dataset = tf.data.TFRecordDataset( [ Settings.EVAL_DATASET_FILE ] )
eval_dataset = eval_dataset.prefetch(10000)
eval_dataset = eval_dataset.batch( Settings.BATCH_SIZE )
eval_dataset = eval_dataset.map( example_parse_function )

model = create_model(item_labels, customer_labels)

# TODO: check loss function...
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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

# TF 2.3: Requires validation_steps. It seems a bug, as documentation says it can be None for TF datasets, but
# with None it throws exception
model.fit(train_dataset, 
          epochs=Settings.N_EPOCHS,
          callbacks=[tensorboard_callback, cp_callback], 
          validation_data=eval_dataset,
          validation_steps=n_eval_batches)
