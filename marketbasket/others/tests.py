
import numpy as np

#x = np.array( [ [0,0] , [1,1], [2,2] ] )
x = np.array( [] , dtype='str')
i = np.array( [0, 2] )
v = np.array( [99,99] )
x = np.insert( x , 0 , v , axis = 0)
print(x)

