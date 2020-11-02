import random
from settings import Settings

# Fix seed to  get reproducible datasets
random.seed(1)

# Shuffle transactions, to be sure there are no sequences with same features together
lines = open(Settings.TRANSACTIONS_FILE).readlines()
random.shuffle(lines)
open(Settings.TRANSACTIONS_FILE, 'w').writelines(lines)
