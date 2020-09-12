
class Settings:

    # Max number of items to handle
    N_MAX_ITEMS = 50

    # Max number customers to handle. If zero, customer code will not be trained
    N_MAX_CUSTOMERS = 10

    # Ratio (1 = 100%) of samples to use for evaluation
    EVALUATION_RATIO = 0.15

    # Batch size
    BATCH_SIZE = 64

    # Epochs to train
    N_EPOCHS = 5
    
    # Sequential model?
    SEQUENTIAL = True

    # Sequence length?
    SEQUENCE_LENGTH = 5

    # Sequence - Embbeding dimension
    EMBEDDING_DIM = 12