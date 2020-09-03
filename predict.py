import tensorflow as tf
from labels import Labels
import numpy as np
from operator import itemgetter

product_labels = Labels.load()

model = tf.keras.models.load_model('model/exported_model')
model.summary()

txt_item_codes = ['21131' , '21730' ]
item_indices = [ product_labels.indices[txt_item_code] for txt_item_code in txt_item_codes ]

n_items = len(product_labels.labels)
input_data = np.zeros((1, n_items))
input_data[0][item_indices] = 1.0
#print(input_data)

result = model.predict(input_data)[0]
print( result[ product_labels.indices['3979'] ] )

N_MAX = 10
indexed_result = list(enumerate(result))
top_n = sorted(indexed_result, key=itemgetter(1))[-N_MAX:]
#print(top_n)
top_indices = list(reversed([i for i, v in top_n]))
print(top_indices)

max_results = [ ( product_labels.labels[idx] , result[idx] ) for idx in top_indices ]
print(max_results)
