import requests

data = {
    "age": 57,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 236,
    "fbs": 0,
    "restecg": 0,
    "thalach": 174,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 1,
    "thal": 1
}

response = requests.post("http://10.26.65.217:5000/predict", json=data)
print(response.json())