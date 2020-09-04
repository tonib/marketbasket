
class Settings:

    # Max number of items to handle
    N_MAX_ITEMS = 50

    # Max number customers to handle. If zero, customer code will not be trained
    N_MAX_CUSTOMERS = 10

    # Ratio (1 = 100%) of samples to use for evaluation
    EVALUATION_RATIO = 0.15

    # Batch size
    BATCH_SIZE = 128

    # Epochs to train
    N_EPOCHS = 30

    # Items labels file path
    ITEM_LABELS_FILE = 'data/itemlabels.txt'
    
    # Customer labels file path
    CUSTOMER_LABELS_FILE = 'data/customerlabels.txt'

    # Train dataset file path
    TRAIN_DATASET_FILE = 'data/dataset_train.tfrecord'

    # Train dataset file path
    EVAL_DATASET_FILE = 'data/dataset_eval.tfrecord'
