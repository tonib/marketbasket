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

url = 'http://localhost:8501/v1/models/basket:predict'

start = time.time()
response = requests.post(url, data=data)
print("Total time:", time.time() - start)
print(response.text)