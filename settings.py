from enum import Enum

class ModelType(Enum):
    DENSE = "dense"
    RNN = "rnn"
    CONVOLUTIONAL = "convolutional"
    GPT = "gpt"

class Settings:

    def __init__(self):

        # Max number of items to handle
        self.N_MAX_ITEMS = 100

        # Max number customers to handle. If zero, customer code will not be trained
        self.N_MAX_CUSTOMERS = 100

        # Ratio (1 = 100%) of samples to use for evaluation
        self.EVALUATION_RATIO = 0.15

        # Batch size
        self.BATCH_SIZE = 64

        # Epochs to train
        self.N_EPOCHS = 15
        
        # Use class weights to correct labels imbalance?
        self.CLASS_WEIGHT = False

        # Model type
        self.MODEL_TYPE: ModelType = ModelType.CONVOLUTIONAL

        # Sequence length?
        self.SEQUENCE_LENGTH = 16

        # Sequence - Items embeding dimension
        self.ITEMS_EMBEDDING_DIM = 128

        # Sequence - Customers embeding dimension
        self.CUSTOMERS_EMBEDDING_DIM = 64

        # Transactions file path
        self.TRANSACTIONS_FILE = 'data/transactions.txt'


# Global variable. TODO: It should not be a global variable
settings = Settings()

