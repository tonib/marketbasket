import tensorflow as tf
from settings import Settings
from labels import Labels
from dataset import DataSet

item_labels = Labels.load(Labels.ITEM_LABELS_FILE)
customer_labels = Labels.load(Labels.CUSTOMER_LABELS_FILE)
DataSet.setup_feature_keys(item_labels, customer_labels)

# Define train dataset
train_dataset = DataSet.load_debug_train_dataset()

# 2 7 8 9 -> 2
# 3979 3243 4566 140335 -> 3979
# [UNKNOWN] 140335 4565 4566 3979 3243

for idx, record in enumerate(train_dataset.take(30)):
    print(record)
    #print(record[0]['input_items_idx'].shape)
