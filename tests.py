
import tensorflow as tf
from labels import Labels

@tf.function
def count_non_equal(batch, value):
    elements_equal_to_value = tf.not_equal(batch, value)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints, axis=1)
    return count

@tf.function
def remove_not_found_index(batch_item_indices, not_found_index):
    # Count non -1's on each row
    found_counts_per_row = count_non_equal(batch_item_indices, not_found_index)
    print("found_counts_per_row", found_counts_per_row)

    # Get non -1 values batch_item_indices from flat values
    flat_values = batch_item_indices.flat_values
    mask = tf.not_equal( flat_values , not_found_index )
    print("mask", mask )
    flat_found_indices = tf.boolean_mask( flat_values , mask )
    print("flat_found_indices", flat_found_indices )
    return tf.RaggedTensor.from_row_lengths( flat_found_indices , found_counts_per_row )

@tf.function
def preprocess_items(batch_item_labels: tf.RaggedTensor):

    # Define lookup tables
    not_found_index = -1
    item_labels_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            Labels.ITEM_LABELS_FILE, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" "), not_found_index)

    # Do lookups item label -> index, -1 if not found
    batch_item_indices = tf.ragged.map_flat_values(item_labels_lookup.lookup, batch_item_labels)
    print( "batch_item_indices", batch_item_indices )
    # Remove -1's:
    batch_item_indices = remove_not_found_index(batch_item_indices, not_found_index)

    # Remove duplicated items
    # TODO: UNIMPLEMENTED. tf.unique works only with 1D dimensions...
    # batch_item_indices = tf.map_fn(lambda x: tf.unique(x), batch_item_indices.to_tensor(-1) )
    # print("unique", tf.unique(batch_item_indices))

    return batch_item_indices

@tf.function
def prepreprocess_customers(batch_customer_labels: tf.Tensor):
    # Define lookup tables
    not_found_index = -1
    customer_labels_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            Labels.CUSTOMER_LABELS_FILE, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" "), not_found_index)

    # Get customer label
    batch_customer_indices = customer_labels_lookup.lookup(batch_customer_labels)
    batch_customer_indices = tf.cast( batch_customer_indices , tf.int32 )

    # Get the "UNKNOWN" customer index
    unknown_customer_index = customer_labels_lookup.lookup( tf.constant(Labels.UNKNOWN_LABEL, dtype=tf.string) )
    unknown_customer_index = tf.cast( unknown_customer_index , tf.int32 )

    # Replace -1 by "UNKNOWN" index
    update_indices = tf.where( tf.math.equal(batch_customer_indices, not_found_index) )
    batch_customer_indices = tf.tensor_scatter_nd_update( batch_customer_indices, update_indices, 
        tf.repeat( unknown_customer_index , tf.size( update_indices ) ) )
    return batch_customer_indices


@tf.function
def test( batch_item_labels: tf.RaggedTensor, batch_customer_labels: tf.Tensor, sequence_length: int ) -> tf.Tensor:
    return ( preprocess_items(batch_item_labels) , prepreprocess_customers(batch_customer_labels) )


item_labels = tf.ragged.constant( [ ['3979', '4565'], ['3979'] , [] , ['achilipu', 'achilipu'] , ['3979', 'achilipu'] ] )
customer_labels = tf.constant( [ '12626', 'achilipu' , '8333' , 'arriquitaun' ] )
print("result", test(item_labels, customer_labels, 3) )

