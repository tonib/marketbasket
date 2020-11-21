from enum import Enum
import argparse
import json

class ModelType(Enum):
    """ Available model types """
    DENSE = "dense"
    RNN = "rnn"
    CONVOLUTIONAL = "convolutional"
    GPT = "gpt"


class Settings:
    """ Model, data and train settings """

    def __init__(self):
        
        # Read config JSON file, if it was specified
        cmd_line_options = self._parse_cmd_line()
        if cmd_line_options.configfile != None:
            settings_json = self._load_config_file(cmd_line_options.configfile)
        else:
            settings_json = {}

        # Setup configuration
        # Max number of items to handle
        self.n_max_items = self._read_setting( settings_json, 'n_max_items' , int , 100 )

        # Max number customers to handle. If zero, customer code will not be trained
        self.n_max_customers = self._read_setting( settings_json, 'n_max_customers' , int , 100 )

        # Ratio (1 = 100%) of samples to use for evaluation
        self.evaluation_ratio = self._read_setting( settings_json, 'evaluation_ratio' , float , 0.15 )

        # Batch size
        self.batch_size = self._read_setting( settings_json, 'batch_size' , int , 64 )

        # Epochs to train
        self.n_epochs = self._read_setting( settings_json, 'n_epochs' , int , 15 )
        
        # Use class weights to correct labels imbalance?
        self.class_weight = self._read_setting( settings_json, 'class_weight' , bool , False )

        # Model type
        self.model_type = self._read_setting( settings_json, 'model_type' , ModelType , ModelType.CONVOLUTIONAL )

        # Sequence length?
        self.sequence_length = self._read_setting( settings_json, 'sequence_length' , int , 16 )

        # Sequence - Items embeding dimension
        self.items_embedding_dim = self._read_setting( settings_json, 'items_embedding_dim' , int , 128 )

        # Sequence - Customers embeding dimension
        self.customers_embedding_dim = self._read_setting( settings_json, 'customers_embedding_dim' , int , 64 )

        # Transactions file path
        self.transactions_file = self._read_setting( settings_json, 'transactions_file' , str , 'data/transactions.txt' )


    def _read_setting(self, settings_json: dict, key: str, value_type: type, default_value: object) -> object:
        if key in settings_json:
            return value_type( settings_json[key] )
        return default_value

    def _parse_cmd_line(self) -> object:
        """ Parse command line and return options """

        parser = argparse.ArgumentParser(description='Market basket analysis')
        parser.add_argument('--configfile', metavar='file_path', type=str, 
            help='Path to JSON file with configuration. If not specified a default configuration will be used')
        return parser.parse_args()


    def _load_config_file(self, config_file_path: str):
        with open( config_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            return json.loads(json_text)


    def print_summary(self):
        print("Settings:")
        print("-----------------------------------------")
        for key in self.__dict__:
            print(key + ': ' + str(self.__dict__[key]))
        print("-----------------------------------------")

# Global variable. TODO: It should not be a global variable
settings = Settings()
settings.print_summary()

exit()
