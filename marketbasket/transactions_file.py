from .settings import settings
from .transaction import Transaction

class TransactionsFile:
    """ Interface to read/write transactions from csv file """

    def __init__(self, path: str, mode: str):
        """ Open a transactions file to read or write
            Args:
                path: Transactions file path
                mode: 'r' to read or 'w' to write
        """
        self._file = open(path, mode)

        # Read/write (column) names
        if mode == 'r':
            # Read column titles
            txt_line = self._file.readline().strip()
            self.feature_names = txt_line.split(";")
        else:
            # Write column titles
            self.feature_names = settings.features.features_names
            self._file.write(";".join(self.feature_names) + "\n")
        
    def write(self, transaction: Transaction):
        """ Writes transaction to file """
        feature_values = []
        for feature_name in self.feature_names:
            feature_value = transaction[feature_name]
            if isinstance(feature_value, list):
                feature_value = " ".join(feature_value)
            feature_values.append( feature_value )

        txt_features = ";".join( [str(feature_value) for feature_value in feature_values] )
        self._file.write( txt_features + "\n" )

    def _read(self, line_txt: str) -> Transaction:
        """ Reads transaction from file """

        line_txt = line_txt.strip()

        # Read file row values
        column_values = line_txt.split(";")
        if len(column_values) < len(self.feature_names):
            raise Exception("Wrong number of columns in line " + str(column_values) + ". Column titles:" + str(self.feature_names) )

        # Map values to with feature names
        features_dict = {}
        for idx in range( len(self.feature_names) ):
            features_dict[ self.feature_names[idx] ] = column_values[idx]

        return Transaction(features_dict)

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._file.close()

    def __iter__(self) -> Transaction:
        for txt_line in self._file:
            yield self._read(txt_line)

    @staticmethod
    def train_dataset_path() -> str:
        """ Returns raw train transactions file path """
        return settings.get_data_path('train_transactions.csv')