from .settings import settings
from typing import List, Tuple, Dict
from .labels import Labels
from .feature import Feature
import tensorflow as tf

class Transaction:
    
    def __init__(self, feature_values: Dict[str, str] = None):

        if not feature_values:
            self._features = {}
            return

        # Features values (key=feature name, value=feature value)
        self._features = feature_values

        # Split sequence features
        for feature in settings.features:
            if feature.sequence:
                self._features[feature.name] = self._features[feature.name].split(' ')
    
    def __getitem__(self, feature_name: str) -> object:
        """ Returns value for a feature name """
        return self._features[feature_name]

    def __setitem__(self, feature_name, value):
        """ Assign a feature value """
        self._features[feature_name] = value

    @property
    def customer_label(self) -> str:
        """ Customer label (TODO: Remove this, it's just to keep compatibly with previous system) """
        return self._features['CliCod']

    @customer_label.setter
    def customer_label(self, value: str):
        """ Customer label (TODO: Remove this, it's just to keep compatibly with previous system) """
        self._features['CliCod'] = value

    @property
    def item_labels(self) -> List:
        """ Item labels/indices for this transaction """
        return self._features[settings.features.item_label_feature]

    @item_labels.setter
    def item_labels(self, value: List):
        """ Item labels/indices for this transaction """
        self._features[settings.features.item_label_feature] = value

    def sequence_length(self) -> int:
        """ Returns the items sequence length in this transaction """
        return len(self.item_labels)

    def get_slice(self, start_idx: int, end_idx: int) -> 'Transaction':
        """ Returns transaction with an slice of the items sequence in this transaction"""
        result = Transaction()
        for feature_name in self._features:
            if settings.features[feature_name].sequence:
                result[feature_name] = self._features[feature_name][start_idx:end_idx]
            else:
                # Keep non sequence features as they are
                result[feature_name] = self._features[feature_name]
        return result

    def __repr__(self):
        return str(self._features)

    def replace_labels_by_indices(self) -> 'Transaction':
        """ Returns a copy of this transaction with labels replaced by its indices """
        result = Transaction()
        feature: Feature
        for feature in settings.features:
            feature_value = self._features[feature.name]
            if feature.sequence:
                feature_value = feature.labels.labels_indices(feature_value)
            else:
                feature_value = feature.labels.label_index(feature_value)
            result[feature.name] = feature_value
        return result

    def replace_indices_by_labels(self) -> 'Transaction':
        """ Returns a copy of this transaction with indices replaced by its labels """
        result = Transaction()
        feature: Feature
        for feature in settings.features:
            feature_value = self._features[feature.name]
            if feature.sequence:
                feature_value = feature.labels.indices_to_labels(feature_value)
            else:
                feature_value = feature.labels.index_label(feature_value)
            result[feature.name] = feature_value
        return result

    def remove_unknown_item_indices(self) -> 'Transaction':
        # Get unknown item indices
        unknown_item_indices = []
        for idx, item_idx in enumerate(self.item_labels):
            if item_idx < 0:
                unknown_item_indices.append(idx)

        if len(unknown_item_indices) == 0:
            # No unknown items
            return self

        # Remove unknown item positions from all sequence features
        unknown_item_indices = reversed(unknown_item_indices)
        result = Transaction()
        feature: Feature
        for feature in settings.features:
            feature_value = self[feature.name]
            if feature.sequence:
                feature_value = list(feature_value) # Clone
                for idx in unknown_item_indices:
                    del feature_value[idx]
            result[feature.name] = feature_value
        return result

    def to_example_features(self) -> Dict[str, tf.train.Feature]:
        """ Returns transaction features as tf.train.Feature """
        example_features = {}
        for feature in settings.features:
            feature_value = self._features[feature.name]
            if not feature.sequence:
                # tf.train.*List requires an iterable
                feature_value = [feature_value]
            example_features[feature.name] = tf.train.Feature( int64_list=tf.train.Int64List( value=feature_value ) )
        return example_features

    # TODO: Remove this
    def to_net_inputs(self, item_labels: Labels, customer_labels: Labels) -> Tuple[ List[int], int ]:

        # Get item indices
        item_indices = []
        for item_label in self.item_labels:
            if item_labels.contains(item_label):
                item_indices.append( item_labels.label_index(item_label) )
        
        # Get customer index to feed
        if not customer_labels.contains(self.customer_label):
            customer_label = Labels.UNKNOWN_LABEL
        else:
            customer_label = self.customer_label
        
        return ( item_indices , customer_labels.label_index(customer_label) )

    # TODO: Remove this
    @staticmethod
    def to_net_inputs_batch(transactions: List['Transaction']):
        batch_item_labels = []
        batch_customer_labels = []
        for transaction in transactions:
            batch_item_labels.append( transaction.item_labels )
            batch_customer_labels.append( transaction.customer_label )
        return ( batch_item_labels , batch_customer_labels )

    # TODO: Remove this
    @staticmethod
    def from_labels(item_labels: List[str], customer_label: str):
        transaction = Transaction()
        transaction.item_labels = item_labels
        transaction.customer_label = customer_label
        return transaction
