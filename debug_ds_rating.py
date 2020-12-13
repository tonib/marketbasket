import marketbasket.dataset as dataset

"""
    Script to debug rating model dataset
"""

# Define train dataset
dataset = dataset.get_dataset(True, False, True)

for idx, record in enumerate(dataset.take(30)):
    print(record)
