import requests
import base64
import os
import json

#os.environ['NO_PROXY'] = '127.0.0.1'

def make_request(prompt, image, parameters):

    app_dict = {
        "instances" : [
            {
                'prompt' : text,
                'image' : image,
            },
            {
                'prompt' : text,
                'image' : image,
            },
        ]
    }
    app_dict['instances'][0]['parameters'] = {}
    app_dict['instances'][1]['parameters'] = {}
    app_dict['instances'][0]['parameters'] = parameters
    app_dict['instances'][1]['parameters'] = parameters
    r = requests.post("http://127.0.0.1/predict", json=app_dict)
    print(r.status_code, r.reason)
    response = json.loads(r.text)
    with open('response.json', 'w') as f:
        json.dump(response, f)

text = "a photo of (eggs) and (((bacon))) on a frying pan"

## Txt2Img

parameters = {
    'type' : "txt2img",
    'ddim_steps' : 30,
    'scale' : 7.5,
    'n_samples' : 2,
    'n_itter' : 2
}

image = None

#make_request(text, image, parameters)

## Img2Img

parameters = {
    'type' : 'img2img',
    'ddim_steps' : 20,
    'scale' : 7.5,
    'n_samples' : 2,
    'n_itter' : 2,
    'strength' : 0.85
}

with open('../images/ddlm_2.png', "rb") as image_file:
    ddlm_image = base64.b64encode(image_file.read())

make_request(text, ddlm_image.decode('utf-8'), parameters)