import marketbasket.dataset as dataset

"""
    Script to debug datasets
"""

# Define train dataset
dataset = dataset.get_dataset(True, True)

for idx, record in enumerate(dataset.take(30)):
    print(record)
