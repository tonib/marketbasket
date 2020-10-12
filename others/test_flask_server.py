import requests
import time

data = '''
    {
        "signature_name":"predict", 
        "inputs": { 
            "customer_label": "5909", 
            "item_labels": ["21131", "21177", "4565", "4682"], 
            "n_results": 10 
        }
    }
'''

# data = '''
#     {
#         "signature_name":"predict", 
#         "inputs": { 
#             "customer_label": "4", 
#             "item_labels": ["notexists"], 
#             "n_results": 10 
#         }
#     }
# '''


# Damn slow in Windows (see https://stackoverflow.com/questions/59506097/python-requests-library-is-very-slow-on-windows)
# url = 'http://localhost:5001/predict'
url = 'http://127.0.0.1:5001/v1/models/basket:predict'


for _ in range(5):
    start = time.time()
    response = requests.post(url, data=data)
    print("Total time:", time.time() - start)
    print(response.text)
