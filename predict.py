import tensorflow as tf
from labels import Labels
from typing import List, Tuple
from transaction import Transaction
from model import pad_sequence # Required to load the model...

class Prediction:

    def __init__(self, model = None):

        self.item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
        self.customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)

        if model:
            self.model: tf.keras.Model = model
        else:
            self.model: tf.keras.Model = tf.keras.models.load_model('model/exported_model')
            self.model.summary()

        self.n_items = len(self.item_labels.labels)
        self.n_customers = len(self.customer_labels.labels)


    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def post_process_items(batch_items_indices: tf.Tensor) -> tf.Tensor:

        # Define reverse lookup tables (tem index -> item string label)
        # Define lookup tables
        item_indices_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            Labels.ITEM_LABELS_FILE, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
            tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, delimiter=" "), "")

        # Do lookup
        return item_indices_lookup.lookup( tf.cast(batch_items_indices, tf.int64) )


    @staticmethod
    @tf.function( input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                   tf.TensorSpec(shape=[], dtype=tf.int64)] )
    def _top_predictions_tensor(results, n_results):

        # Get most probable item indices
        sorted_indices = tf.argsort(results, direction='DESCENDING')
        top_indices = sorted_indices[:,0:n_results]
        top_probabilities = tf.gather(results, top_indices, batch_dims=1)

        # Convert item indices to item labels
        #print("top_indices", top_indices)
        top_item_labels = Prediction.post_process_items(top_indices)

        return top_item_labels, top_probabilities


    @staticmethod
    @tf.function( input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), 
                                   tf.TensorSpec(shape=[None, None], dtype=tf.float32)] )
    def _remove_input_items_from_prediction(batch_item_indices, result):
        # result is a ragged with input indices for each batch row. Ex [ [0] , [1, 2] ]
        batch_item_indices = batch_item_indices.to_tensor(-1) # -> [ [0,-1] , [1, 2] ]
        #print(batch_item_indices, batch_item_indices.shape[0])

        # Convert batch_item_indices row indices to (row,column) indices
        row_indices = tf.range(0, tf.shape(batch_item_indices)[0], dtype=tf.int64) # -> [ 0, 1 ]
        row_indices = tf.repeat( row_indices, [tf.shape(batch_item_indices)[1]] ) # -> [ 0, 0, 1, 1 ]
        #print(">>>", batch_item_indices)
        batch_item_indices = tf.reshape( batch_item_indices , shape=[-1] ) # -> [ 0, -1, 1, 2 ]
        batch_item_indices = tf.stack( [row_indices, batch_item_indices], axis = 1 ) # -> [ [0,0] , [0,-1], [1,1], [1,2] ]

        # batch_item_indices.to_tensor(-1) added -1's to pad the matrix. Remove these indices
        # Needed according to tf.tensor_scatter_nd_update doc. (it will fail in CPU execution, if there are out of bound indices)
        # Get indices without -1's:
        gather_idxs = tf.where( batch_item_indices[:,1] != -1 ) # -> [[0], [2], [3]]
        batch_item_indices = tf.gather_nd(batch_item_indices, gather_idxs) # -> [ [0,0] , [1,1], [1,2] ]

        # To remove input indices, we will set a probability -1 in their indices
        updates = tf.repeat(-1.0, tf.shape(batch_item_indices)[0]) # -> [ -1, -1, -1 ]

        # Assign -1's to the input indices:
        return tf.tensor_scatter_nd_update( result, batch_item_indices, updates)


    @staticmethod
    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.int64)])
    def count_non_equal(batch: tf.RaggedTensor, value: tf.Tensor) -> tf.Tensor:
        """ Returns the count of elements in 'batch'' distincts to 'value' on each batch row """
        elements_equal_to_value = tf.not_equal(batch, value)
        as_ints = tf.cast(elements_equal_to_value, tf.int64)
        count = tf.reduce_sum(as_ints, axis=1)
        return count


    @staticmethod
    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.int64)])
    def remove_not_found_index(batch_item_indices: tf.RaggedTensor, not_found_index: tf.Tensor) -> tf.RaggedTensor:
        # Count non -1's on each row
        found_counts_per_row = Prediction.count_non_equal(batch_item_indices, not_found_index)
        #print("found_counts_per_row", found_counts_per_row)

        # Get non -1 values batch_item_indices from flat values
        flat_values = batch_item_indices.flat_values
        mask = tf.not_equal( flat_values , not_found_index )
        #print("mask", mask )
        flat_found_indices = tf.boolean_mask( flat_values , mask )
        #print("flat_found_indices", flat_found_indices )
        return tf.RaggedTensor.from_row_lengths( flat_found_indices , found_counts_per_row )


    @staticmethod
    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.string)])
    def preprocess_items(batch_item_labels: tf.RaggedTensor):
        #print("batch_item_labels ->", batch_item_labels)

        # Define lookup tables (item string label -> item index)
        not_found_index = -1
        item_labels_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
                Labels.ITEM_LABELS_FILE, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
                tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" "), not_found_index)

        # Do lookups item label -> index, -1 if not found
        batch_item_indices = tf.ragged.map_flat_values(item_labels_lookup.lookup, batch_item_labels)
        #print( "batch_item_indices", batch_item_indices )

        # Remove -1's:
        batch_item_indices = Prediction.remove_not_found_index(batch_item_indices, not_found_index)
        #print( "batch_item_indices >>>", batch_item_indices )

        # Remove duplicated items
        # TODO: UNIMPLEMENTED. tf.unique works only with 1D dimensions...
        # batch_item_indices = tf.map_fn(lambda x: tf.unique(x), batch_item_indices.to_tensor(-1) )
        # print("unique", tf.unique(batch_item_indices))

        return batch_item_indices


    @staticmethod
    @tf.function
    def prepreprocess_customers(batch_customer_labels: tf.Tensor):
        # Define lookup tables
        not_found_index = -1
        customer_labels_lookup = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
                Labels.CUSTOMER_LABELS_FILE, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
                tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=" "), not_found_index)

        # Get customer label
        batch_customer_indices = customer_labels_lookup.lookup(batch_customer_labels)

        # Get the "UNKNOWN" customer index
        unknown_customer_index = customer_labels_lookup.lookup( tf.constant(Labels.UNKNOWN_LABEL, dtype=tf.string) )

        # Replace -1 by "UNKNOWN" index
        update_indices = tf.where( tf.math.equal(batch_customer_indices, not_found_index) )
        batch_customer_indices = tf.tensor_scatter_nd_update( batch_customer_indices, update_indices, 
            tf.repeat( unknown_customer_index , tf.size( update_indices ) ) )
        return batch_customer_indices


    @tf.function
    def _run_model_and_postprocess(self, batch_item_indices , batch_customer_indices, n_results):

        # Run the model
        batch = ( batch_item_indices , batch_customer_indices )
        result = self.model( batch )

        # Set result[ batch_item_indices ] = -1.0:
        #print("batch_item_indices >>>***", batch_item_indices)
        result = Prediction._remove_input_items_from_prediction( batch_item_indices, result )

        # Get most probable n results
        return Prediction._top_predictions_tensor(result, n_results)


    @tf.function
    def _run_model_filter_empty_sequences(self, batch_item_indices, batch_customer_indices, n_results):

        # Check if there are empty sequences    
        sequences_lenghts = batch_item_indices.row_lengths()
        non_empty_seq_count = tf.math.count_nonzero(sequences_lenghts)
        n_sequences = tf.shape( sequences_lenghts, tf.int64 )[0]

        #print(">>>", non_empty_seq_count, n_results)
        if non_empty_seq_count >= n_sequences:
            # There are no empty sequences. Run the model
            return self._run_model_and_postprocess(batch_item_indices , batch_customer_indices, n_results)
        else:
            # Model will fail if a sequence is empty, and it seems it's the expected behaviour: Do not feed empty sequences
            # Get non empty sequences mask
            non_empty_mask = tf.math.greater( sequences_lenghts , 0 )

            # Get non empty sequences
            non_empty_sequences: tf.RaggedTensor = tf.ragged.boolean_mask( batch_item_indices , non_empty_mask )
            non_empty_customers = tf.boolean_mask( batch_customer_indices , non_empty_mask )
            
            # Run model
            label_predictions, probs_predictions = self._run_model_and_postprocess(non_empty_sequences , non_empty_customers, n_results)

            # Merge real predictions with empty predictions for empty sequences:
            indices = tf.where(non_empty_mask)
            final_shape = [n_sequences, n_results]
            label_predictions = tf.scatter_nd( indices , label_predictions , final_shape )
            #print(label_predictions)
            probs_predictions = tf.scatter_nd( indices , probs_predictions , final_shape )
            #print(probs_predictions)
            return (label_predictions, probs_predictions)
        

    @tf.function(input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.string), 
                                  tf.TensorSpec(shape=[None], dtype=tf.string),
                                  tf.TensorSpec(shape=[], dtype=tf.int64)])
    def _run_model_prediction(self, batch_item_labels, batch_customer_labels, n_results):

        # Convert labels to indices
        #print(">>> batch_item_labels", batch_item_labels)
        batch_item_indices = Prediction.preprocess_items(batch_item_labels)
        #print(">>> batch_item_indices", batch_item_indices)
        #print(">>> batch_customer_labels", batch_customer_labels)
        batch_customer_indices = Prediction.prepreprocess_customers(batch_customer_labels)
        #print(">>> batch_customer_indices", batch_customer_indices)
        
        # Run the model
        return self._run_model_filter_empty_sequences(batch_item_indices, batch_customer_indices, n_results)


    def predict_batch(self, transactions: List[Transaction], n_items_result: int) -> List:

        # Setup batch
        batch = Transaction.to_net_inputs_batch(transactions)

        #print("*** batch", batch)
        batch = ( tf.ragged.constant(batch[0], dtype=tf.string) , tf.constant(batch[1]) )
        #print("*** batch", batch)

        results = self._run_model_prediction( batch[0] , batch[1], n_items_result )
        #print("raw", results)

        results = ( results[0].numpy() , results[1].numpy() )
        return results


    def predict_single(self, transaction: Transaction, n_items_result: int) -> List[ Tuple[str, float] ]:
        results = self.predict_batch( [ transaction ] , n_items_result )
        return ( results[0][0] , results[1][0] )
