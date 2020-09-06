
#import tensorflow as tf
import numpy as np

x = np.array( [ 3 , 1 , 2 ] )
i = np.argsort( x )
print(i)
print(i.shape[0])
print(len(i))

i = i[-2:]
print(i)

i = np.flip(i)
print(i)

