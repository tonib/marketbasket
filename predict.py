import tensorflow as tf
from labels import Labels
import numpy as np
from operator import itemgetter
from typing import List, Tuple
from settings import Settings
from tensorflow.python.framework.ops import disable_eager_execution
from transaction import Transaction

class Prediction:

    def __init__(self):

        disable_eager_execution()

        self.item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
        self.customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)

        self.model = tf.keras.models.load_model('model/exported_model')
        self.model.summary()

        self.n_items = len(self.item_labels.labels)
        self.n_customers = len(self.customer_labels.labels)

    def _transaction_to_inputs(self, transaction: Transaction) -> Tuple[ dict , List[int] ]:

        # Get multi-hot with item indices to feed
        item_indices = []
        multihot_item_indices = np.zeros( (self.n_items) )
        for item_label in transaction.item_labels:
            if self.item_labels.contains(item_label):
                item_idx = self.item_labels.label_index(item_label)
                multihot_item_indices[ item_idx ] = 1.0
                item_indices.append( item_idx )
        
        if len(item_indices) == 0:
            return None

        net_inputs = { 'input_items_idx': multihot_item_indices }

        # Gest customer index to feed, as a batch with size 1
        if Settings.N_MAX_CUSTOMERS > 0:
            if not self.customer_labels.contains(transaction.customer_label):
                customer_label = Labels.UNKNOWN_LABEL
            else:
                customer_label = transaction.customer_label
            net_inputs['customer_idx'] = np.zeros( (self.n_customers) )
            net_inputs['customer_idx'][ self.customer_labels.label_index(customer_label) ] = 1.0

        return net_inputs, item_indices

    def _top_predictions(self, result: List[float], item_indices:List[int], n_items_result: int ) -> List[ Tuple[str, float] ]:

        # "Remove" feeded items indices: Set its probabiblity to negative
        result[ item_indices ] = -1.0

        # Get top item labels with its probability
        top_indices = np.argsort( result )
        top_indices = top_indices[0:n_items_result]
        return [ ( self.item_labels.labels[idx] , result[idx] ) for idx in top_indices ]

    def predict_single(self, transaction: Transaction, n_items_result: int) -> List[ Tuple[str, float] ]:

        net_inputs, item_indices = self._transaction_to_inputs(transaction)

        # Create batch size 1
        net_inputs['input_items_idx'] = np.array( [ net_inputs['input_items_idx'] ] )
        if Settings.N_MAX_CUSTOMERS > 0:
            net_inputs['customer_idx'] = np.array( [ net_inputs['customer_idx'] ] )

        result = self.model.predict(net_inputs)

        return self._top_predictions( result[0], item_indices, n_items_result )

    def predict_batch(self, transactions: List[Transaction], n_items_result: int) -> List[ List[ Tuple[str, float] ] ]:

        # Setup batch
        batch_item_indices = []
        batch = { 'input_items_idx': [] }
        if Settings.N_MAX_CUSTOMERS > 0:
            batch['customer_idx'] = []

        # Prepare batch
        for transaction in transactions:
            net_inputs, item_indices = self._transaction_to_inputs(transaction)

            batch['input_items_idx'].append( net_inputs['input_items_idx'] )
            if Settings.N_MAX_CUSTOMERS > 0:
                batch['customer_idx'].append( net_inputs['customer_idx'] )

            batch_item_indices.append( item_indices )

        # Do not mix Python and numpy arrays
        batch['input_items_idx'] = np.array( batch['input_items_idx'] )
        if Settings.N_MAX_CUSTOMERS > 0:
            batch['customer_idx'] = np.array( batch['customer_idx'] )

        #results = self.model.predict(batch)
        # DO NOT DELETE, TO TEST PYTHON DAMN PERFORMANCE (fake prediction)
        #results = np.random.uniform( size=( len(transactions), len(self.item_labels.labels) ) )
        results = np.zeros( shape=( len(transactions), len(self.item_labels.labels) ) )

        # Unpack results
        top_predictions = []
        for idx, result in enumerate(results):
            top_predictions.append( self._top_predictions(result, batch_item_indices[idx], n_items_result) )

        return top_predictions
