from test_flask_server import data, url
import requests

# Tensorflows prediction seems to be non thread safe
# Test concurrency with multiple instances of this script

while True:
    response = requests.post(url, data=data).json()
    if 'error' in response:
        print(response.error)

