import requests
import base64
import os
import json

#os.environ['NO_PROXY'] = '127.0.0.1'

def make_request(image, parameters, questions=[],captions=[]):

    app_dict = {
        "instances" : [
            {
                'image' : image.decode('utf-8'),
                'questions' : questions,
                'captions' : captions
            },
            {
                'image' : image.decode('utf-8'),
                'questions' : questions,
                'captions' : captions
            },
        ]
    }
    app_dict['instances'][0]['parameters'] = {}
    app_dict['instances'][1]['parameters'] = {}
    app_dict['instances'][0]['parameters'] = parameters
    app_dict['instances'][1]['parameters'] = parameters
    r = requests.post("http://127.0.0.1/predict", json=app_dict)
    print(r.status_code, r.reason)
    print(r.text)


with open('../images/blip_test.jpeg', "rb") as image_file:
    image = base64.b64encode(image_file.read())

## CAPTIONING

parameters = {
    'type' : 'captioning',
    'sample' : False,
    'img_size' : 384,
    'num_beams' : 3,
    'max_length' : 20,
    'min_length' : 5
}

make_request(image, parameters)

## QNA

parameters = {
    'type' : 'qna',
    'img_size' : 480,
}

make_request(image, parameters, questions=['where is the woman sitting?', 'where is the woman sitting?'])



