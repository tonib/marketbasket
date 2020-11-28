from .settings import settings
from typing import List, Tuple
from .labels import Labels
from .feature import Feature

class Transaction:
    
    def __init__(self, feature_values: dict = None):

        if not feature_values:
            self._features = {}
            return

        # Features values (key=feature name, value=feature value)
        self._features = feature_values

        # Split sequence features
        for feature in settings.features:
            if feature.sequence:
                self._features[feature.name] = self._features[feature.name].split(' ')
        
        # # First word is the customer code
        # self.customer_label: str = columns[0]
        # del columns[0]

        # # Others are item labels
        # self.item_labels: List[str] = columns

    def __getitem__(self, feature_name: str) -> object:
        """ Returns value for a feature name """
        return self._features[feature_name]

    @property
    def customer_label(self) -> str:
        """ Customer label (TODO: Remove this, it's just to keep compatibly with previous system) """
        return self._features['CliCod']

    @customer_label.setter
    def customer_label(self, value: str):
        """ Customer label (TODO: Remove this, it's just to keep compatibly with previous system) """
        self._features['CliCod'] = value

    @property
    def item_labels(self) -> List[str]:
        """ Item labels for this transaction """
        return self._features[settings.features.item_label_feature]

    @item_labels.setter
    def item_labels(self, value: List[str]):
        """ Item labels (TODO: Remove this, it's just to keep compatibly with previous system) """
        self._features['ArtCod'] = value

    def __repr__(self):
        return self.customer_label + ' ' + ' '.join(self.item_labels)

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

    @staticmethod
    def to_net_inputs_batch(transactions: List['Transaction']):
        batch_item_labels = []
        batch_customer_labels = []
        for transaction in transactions:
            batch_item_labels.append( transaction.item_labels )
            batch_customer_labels.append( transaction.customer_label )
        return ( batch_item_labels , batch_customer_labels )

    @staticmethod
    def from_labels(item_labels: List[str], customer_label: str):
        transaction = Transaction()
        transaction.item_labels = item_labels
        transaction.customer_label = customer_label
        return transaction

    @staticmethod
    def top_items_path() -> str:
        """ Returns top items transactions file path """
        return settings.get_data_path('transactions_top_items.csv')

    @staticmethod
    def eval_dataset_path() -> str:
        """ Returns raw eval transactions file path """
        return settings.get_data_path('eval_transactions.csv')