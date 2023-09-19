import requests
import base64
import os

def make_request(prompt):

    app_dict = {
        "instances" : [
            {
                'user_input' : prompt,
            }
        ]
    }

    r = requests.post("http://127.0.0.1/predict", json=app_dict)
    print(r.text)

make_request("Hi, how are you today? What is a black hole?")