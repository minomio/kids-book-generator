import requests

BASE = "http://127.0.0.1:5000/"

prompt = "The birds fly high"

response = requests.get(BASE + "helloworld/"+ prompt)
print(response.json())
