from marketbasket.jsonutils import read_setting
from marketbasket.labels import Labels
from typing import List, Union
import marketbasket.settings as settings

class Feature:
    """ A model input/output feature configuration.
        Currently are expected to be string labels
    """

    def __init__(self, config: dict, sequence: bool):
        """ Parse a feature from config file
            Args:
                config  Parsed JSON for this feature
                sequence    True if is a item sequence feature. False if is a transaction feature
        """

        # Feature name
        self.name:str = read_setting(config, 'name', str, Exception("'name' property expected"))

        # If = 0, feature will not be embedded. If > 0, the embedding dimension
        self.embedding_dim:int = read_setting(config, 'embedding_dim', int, 0)

        # Max. number of labels to use as input / predict. All, if == 0.
        self.max_labels = read_setting(config, 'max_labels', int, 0)

        # Belongs to items sequence?
        self.sequence = sequence

        # Label values
        self.labels: Labels = None

    def __repr__(self):
        txt = self.name + ": label"
        if self.embedding_dim > 0:
            txt += " / embedding_dim: " + str(self.embedding_dim)
        if self.max_labels > 0:
            txt += " / max_labels: " + str(self.max_labels)
        return txt
        
    def filter_wrong_labels(self, feature_value: Union[str,List[str]]) -> Union[str,List[str]]:
        """ Remove/replace wrong labels from feature value. If feature
            is the items labels feature, the wrong values will be removed.
            Otherwise, wrong values will be replaced by Labels.UNKNOWN_LABEL
            Args:
                feature_value: Feature value to check. If this feature is a sequence, this should be
                    an array
            Returns: Feature values without wrong labels
        """
        values = feature_value if self.sequence else [feature_value]
        result = []
        is_items_labels = (self.name == settings.settings.features.item_label_feature)
        for value in values:
            if not self.labels.contains(value):
                if not is_items_labels:
                    result.append(Labels.UNKNOWN_LABEL)
            else:
                result.append(value)
        if self.sequence:
            return result
        if len(result) == 0:
            return None
        return result[0]