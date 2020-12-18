import marketbasket.settings as settings
from typing import List, Dict, Iterable
from marketbasket.feature import Feature
from marketbasket.jsonutils import read_setting
from marketbasket.labels import Labels
import marketbasket.dataset as dataset
import copy

class FeaturesSet:
    """ Dataset features """

    def __init__(self, config:dict):
        """ Parses features in configuration file
            Args:
                config: Parsed JSON from configuration file
        """
        
        if config == None:
            return

        # All features. Order here is important
        self._features: Dict[str, Feature] = {}
        # Features related to transaction
        self._transaction_features: Dict[str, Feature] = self._create_features(False, config['transaction_features'])
        # Features related to items sequence
        self._items_sequence_features: Dict[str, Feature] = self._create_features(True, config['items_features'] )
        
        # TODO: Change name of self.item_label_feature to self.items_feature_name, or something like that
        # Store the item labels feature name
        self.item_label_feature:str = read_setting(config, 'item_label_feature', str, None)
        if not self.item_label_feature in self._items_sequence_features:
            raise Exception(self.item_label_feature + " feature not found in 'items_features'")
        
        # Item labels feature must to be embedded: This restriction is done to be sure sequences will be masked
        # See Feature.encode_input()
        if self.items_sequence_feature().embedding_dim <= 0:
            raise Exception("Items labels MUST to have embedding_dim > 0")

        # TODO: Remove self.item_label_index
        # Store items sequence feature index
        for index, feature in enumerate(self):
            if feature.name == self.item_label_feature:
                self.item_label_index = index
                break

    def rating_model_features(self) -> 'FeaturesSet':
        # Do a shallow copy
        rating_features = copy.copy(self)
        rating_features._features = copy.copy(rating_features._features)
        rating_features._transaction_features = copy.copy(rating_features._transaction_features)

        # Add a new transaction feature for item to rate
        item_to_rate_feature = copy.copy(rating_features.items_sequence_feature())
        item_to_rate_feature.name = dataset.ITEM_TO_RATE
        item_to_rate_feature.sequence = False
        
        rating_features._features[item_to_rate_feature.name] = item_to_rate_feature
        rating_features._transaction_features[item_to_rate_feature.name] = item_to_rate_feature

        return rating_features

    def _create_features(self, sequence: bool, features_config: List) -> Dict[str, Feature]:
        """ Read block of features from config.
            Args:
                sequence: True -> item sequence features, False -> whore transaction features)
                features_config: List with JSON features configuration
            Returns:
                Dictionary with features names/features read
        """
        features_set = {}
        for feature_config in features_config:
            if not read_setting(feature_config, 'ignore', bool, False):
                feature = Feature(feature_config, sequence)
                features_set[feature.name] = feature
                self._features[feature.name] = feature
        return features_set

    def items_sequence_feature(self) -> Feature:
        """ Returns the feature for the items sequence """
        return self._features[self.item_label_feature]
        
    @property
    def features_names(self) -> Iterable[str]:
        """ Get features names, ordered by declaration """
        return list(self._features.keys())

    def __getitem__(self, feature_name: str) -> Feature:
        """ Returns feature with a given name """
        return self._features[feature_name]

    def __iter__(self) -> Iterable[Feature]:
        """ Return all features """
        return iter(self._features.values())
        
    def transaction_features(self, except_names=[]) -> Iterable[Feature]:
        """ Return transaction features (non sequence), unordered """
        for feature in self._transaction_features.values():
            if feature.name not in except_names:
                yield feature

    def sequence_features(self) -> Iterable[Feature]:
        """ Return sequence features (features related to items), unordered """
        return self._items_sequence_features.values()

    def __repr__(self) -> str:
        return "Features:\n\t" + "\n\t".join([str(f) for f in self])

    def save_label_files(self):
        """ Saves features label files in data directory """
        for feature in self:
            file_name = "labels_" + feature.name + ".txt"
            feature.labels.save( settings.settings.get_data_path(file_name) )
    
    def load_label_files(self):
        """ Load feature labels from data directory """
        for feature in self:
            file_name = "labels_" + feature.name + ".txt"
            feature.labels = Labels.load( settings.settings.get_data_path(file_name) )
