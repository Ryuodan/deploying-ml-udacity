import requests

#url = 'http://0.0.0.0:5000'

input_data = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14084,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

# req = requests.get(url)#, json=input_data)
# print(req)
# assert req.status_code == 200

# print('Status code:', req.status_code)
#print('Model Inference:', req.json())

url = 'http://0.0.0.0:5000/prediction'

req = requests.post(url, json=input_data)
#print(req.json())
assert req.status_code == 200

print('Status code:', req.status_code)
print('Model Inference:', req.json())