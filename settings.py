
class Settings:

    # Max number of items to handle
    N_MAX_ITEMS = 1000

    # Max number customers to handle. If zero, customer code will not be trained
    N_MAX_CUSTOMERS = 100

    # Ratio (1 = 100%) of samples to use for evaluation
    EVALUATION_RATIO = 0.15

    # Batch size
    BATCH_SIZE = 64

    # Epochs to train
    N_EPOCHS = 10
    
    # Sequential model?
    SEQUENTIAL = True

    # Sequence length?
    SEQUENCE_LENGTH = 10

    # Sequence - Embbeding dimension
    EMBEDDING_DIM = 10