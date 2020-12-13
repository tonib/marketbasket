import marketbasket.dataset as dataset

"""
    Script to debug candidates model datasets
"""

# Define train dataset
dataset = dataset.get_dataset(False, False, True)

for idx, record in enumerate(dataset.take(30)):
    print(record)
