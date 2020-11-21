from enum import Enum

class ModelType(Enum):
    DENSE = "dense"
    RNN = "rnn"
    CONVOLUTIONAL = "convolutional"
    GPT = "gpt"

class Settings:

    def __init__(self):
        self._default_values()
        
    def _default_values(self):
        # Max number of items to handle
        self.n_max_items = 100

        # Max number customers to handle. If zero, customer code will not be trained
        self.n_max_customers = 100

        # Ratio (1 = 100%) of samples to use for evaluation
        self.evaluation_ratio = 0.15

        # Batch size
        self.batch_size = 64

        # Epochs to train
        self.n_epochs = 15
        
        # Use class weights to correct labels imbalance?
        self.class_weight = False

        # Model type
        self.model_type: ModelType = ModelType.CONVOLUTIONAL

        # Sequence length?
        self.sequence_length = 16

        # Sequence - Items embeding dimension
        self.items_embedding_dim = 128

        # Sequence - Customers embeding dimension
        self.customers_embedding_dim = 64

        # Transactions file path
        self.transactions_file = 'data/transactions.txt'


# Global variable. TODO: It should not be a global variable
settings = Settings()

