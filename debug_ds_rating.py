import marketbasket.settings as settings
import marketbasket.dataset as dataset

"""
    Script to debug rating model dataset
"""

settings.settings.features.load_label_files()

# Define train dataset
dataset = dataset.get_dataset(True, False, True)

print("\n\n")
for idx, record in enumerate(dataset.take(30)):
    print("input:", record[0], "\noutput:", record[1], "\n\n")
