from typing import List, Dict, KeysView
from marketbasket.feature import Feature
from marketbasket.jsonutils import read_setting

class FeaturesSet:
    """ Dataset features """

    def __init__(self, config:dict):
        """ Parses features in configuration file
            Args:
                config: Parsed JSON from configuration file
        """
        # TODO: Add default values?
        # All features
        self._features: Dict[str, Feature] = {}
        # Features related to transaction
        self._transaction_features = self._create_features(False, config['transaction_features'])
        # Features related to items sequence
        self._items_sequence_features = self._create_features(True, config['items_features'] )
        
        # Store the item labels feature name
        self.item_label_feature:str = read_setting(config, 'item_label_feature', str, None)
        if not self.item_label_feature in self._items_sequence_features:
            raise Exception(not self.item_label_feature + " feature not found in 'items_features'")

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

    @property
    def features_names(self) -> KeysView[str]:
        """ Get features names """
        return self._features.keys

    def __getitem__(self, feature_name: str) -> Feature:
        """ Returns feature with a given name """
        return self._features[feature_name]

    def __iter__(self):
        return iter(self._features.values())
        
    def __repr__(self):
        return "Features:\n\t" + "\n\t".join([str(f) for f in self])
