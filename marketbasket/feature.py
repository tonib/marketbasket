from marketbasket.jsonutils import read_setting

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

        # Index in transactions file
        self.transactions_file_idx:int = read_setting(config, 'transactions_file_idx', int, 
            Exception("'transactions_file_idx' property expected"))

        # Max. number of labels to use as input / predict. All, if == 0.
        self.max_labels = read_setting(config, 'max_labels', int, 0)

        # Belongs to items sequence?
        self.sequence = sequence

    def __repr__(self):
        txt = self.name + ": " + "transactions_file_idx: " + str(self.transactions_file_idx)
        if self.embedding_dim > 0:
            txt += " / embedding_dim: " + str(self.embedding_dim)
        if self.max_labels > 0:
            txt += " / max_labels: " + str(self.max_labels)
        return txt
        