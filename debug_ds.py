import tensorflow as tf
from settings import Settings
from labels import Labels

item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
N_ITEMS = len( item_labels.labels )

# Feature mappings for training
keys_to_features = {
    'input': tf.io.SparseFeature('sparse_indices', 'sparse_values', tf.float32,  N_ITEMS, False),
    #'sparse_indices': tf.io.VarLenFeature( tf.int64 ),
    'output_label': tf.io.FixedLenFeature([], tf.int64)
}
#print("---------------------- ", keys_to_features['sparse_indices'] )

if Settings.N_MAX_CUSTOMERS > 0:
    keys_to_features['customer'] = tf.io.FixedLenFeature([], tf.int64)

def example_parse_function(proto_batch):
    # Load one example
    parsed_features = tf.io.parse_example(proto_batch, keys_to_features)
    # print("x")
    # print( parsed_features['sparse_indices'] )

    items_input = parsed_features['input']
    #items_input = tf.sparse.to_dense(items_input)
    #items_input = tf.one_hot( parsed_features['sparse_indices'] , N_ITEMS , name='sparse_to_dense_trn' )
    #items_input = parsed_features['sparse_indices']
    #tf.reduce_max()
    if Settings.N_MAX_CUSTOMERS > 0:
        input = { 'input': items_input , 'customer': parsed_features['customer'] }
    else:
        input = items_input
    
    return input, parsed_features['output_label']

# Define train dataset
train_dataset = tf.data.TFRecordDataset( [ DataSet.TRAIN_DATASET_FILE ] )
train_dataset = train_dataset.map( example_parse_function )

for idx, record in enumerate(train_dataset.take(1)):
    print(idx)
    print(record[0])
    print(tf.sparse.to_dense(record[0]))
