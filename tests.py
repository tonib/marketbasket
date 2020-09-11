
import tensorflow as tf
from labels import Labels

def postprocess_item_indices(item_indices: tf.Tensor) -> tf.Tensor:

    # Define lookup tables
    item_indices_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        Labels.ITEM_LABELS_FILE, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, delimiter=" "), "")
    
    return item_indices_lookup.lookup(item_indices)


print( postprocess_item_indices(tf.constant( [2, 3], dtype=tf.int64)) )
