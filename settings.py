from enum import Enum
import argparse
import json
import os

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
        if cmd_line_options.configfile == None and os.path.exists('data/config.json') :
            # Use this as default
            cmd_line_options.configfile = 'data/config.json'

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

        # Model generation directory
        self.model_dir = self._read_setting( settings_json, 'model_dir' , str , 'model' )

        # Train verbose log level
        self.train_log_level = cmd_line_options.trainlog

        # Log level for TF core (C++). This MUST to be executed before import tf
        # See https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
        # See https://github.com/tensorflow/tensorflow/issues/31870
        self.tf_log_level = self._read_setting( settings_json, 'tf_log_level' , str , 'WARNING' )
        if self.tf_log_level == 'WARNING':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        elif self.tf_log_level == 'ERROR':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        

    def _read_setting(self, settings_json: dict, key: str, value_type: type, default_value: object) -> object:
        if key in settings_json:
            return value_type( settings_json[key] )
        return default_value


    def _parse_cmd_line(self) -> object:
        """ Parse command line and return options """
        parser = argparse.ArgumentParser(description='Market basket analysis')
        parser.add_argument('--configfile', metavar='file_path', type=str, 
            help='Path to JSON file with configuration. If not specified and data/config.json exists, this will be used. ' + 
            'Otherwise, a default configuration will be used' )
        parser.add_argument('--trainlog', type=int, nargs='?',
            help='Train verbose log level: 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 1',
            default=1)
        return parser.parse_args()

    def _load_config_file(self, config_file_path: str):
        with open( config_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            return json.loads(json_text)


    def print_summary(self):
        """ Print configuration in stdout """

        print("Settings:")
        print("-----------------------------------------")
        for key in self.__dict__:
            print(key + ': ' + str(self.__dict__[key]))
        print("-----------------------------------------")


    def get_data_path(self, rel_path: str = None) -> str:
        """ Returns data directory if rel_path is None. Path to a file inside data directory otherwise """
        path = os.path.dirname(self.transactions_file)
        if rel_path != None:
            path = os.path.join(path, rel_path)
        return path

    def get_model_path(self, rel_path: str = None) -> str:
        """ Returns model generation directory if rel_path is None. Path to a file inside model directory otherwise """
        path = self.model_dir
        if rel_path != None:
            path = os.path.join(path, rel_path)
        return path

# Global variable. TODO: It should not be a global variable
settings = Settings()

# Set python tf log level
import tensorflow as tf
tf.get_logger().setLevel(settings.tf_log_level)
