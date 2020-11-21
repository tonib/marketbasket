import random
from settings import settings

# Fix seed to  get reproducible datasets
random.seed(1)

# Shuffle transactions, to be sure there are no sequences with same features together
lines = open(settings.transactions_file).readlines()
random.shuffle(lines)
open(settings.transactions_file, 'w').writelines(lines)
