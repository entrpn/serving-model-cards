import requests
import base64
import os

def make_request(prompt, negative_prompt, parameters, init_img=None, mask_img=None):

    app_dict = {
        "instances" : [
            {
                'prompt' : prompt,
                'negative_prompt' : negative_prompt,
            },
            {
                'prompt' : prompt,
                'negative_prompt' : negative_prompt,
            },
        ]
    }
    app_dict['instances'][0]['parameters'] = {}
    app_dict['instances'][1]['parameters'] = {}
    app_dict['instances'][0]['parameters'] = parameters
    app_dict['instances'][1]['parameters'] = parameters

    if init_img is not None:
        app_dict['instances'][0]['init_img'] = init_img.decode('utf-8')
        app_dict['instances'][1]['init_img'] = init_img.decode('utf-8')

    if mask_img is not None: 
        app_dict['instances'][0]['mask_img'] = mask_img.decode('utf-8')
        app_dict['instances'][1]['mask_img'] = mask_img.decode('utf-8')

    r = requests.post("http://127.0.0.1/predict", json=app_dict)
    print(r.text)

# Text2Img

parameters = {
    'type' : 'txt2img',
    'steps' : 35,
    'scale' : 9.5,
    'seed' : 49598694,
    'num_images' : 3,
    'eta' : 0.1,
    'width' : 512,
    'height' : 512,
    'scheduler' : 'EulerAncestralDiscreteScheduler'
}

prompt = "A dog riding a skateboard"
negative_prompt = 'blurry, deformed, purple'

make_request(prompt, negative_prompt, parameters)

# Img2img

parameters = {
    'type' : 'img2img',
    'steps' : 50,
    'scale' : 9.5,
    'seed' : 49598694,
    'num_images' : 2,
    'eta' : 0.1,
    'width' : 512,
    'height' : 512,
    'strength' : 0.75,
    'scheduler' : 'DPMSolverMultistepScheduler'
}

with open('../training-image-segmentation/images/merged.png', "rb") as image_file:
    init_img = base64.b64encode(image_file.read())

prompt = "A selfie of a man by the beach"
negative_prompt = 'blurry, deformed, purple'

make_request(prompt, negative_prompt, parameters, init_img=init_img)

## Inpainting

parameters = {
    'type' : 'inpainting',
    'steps' : 50,
    'scale' : 9.5,
    'seed' : 49598694,
    'num_images' : 1,
    'eta' : 0.1,
    'width' : 512,
    'height' : 512,
    'strength' : 0.75,
    'scheduler' : 'DPMSolverMultistepScheduler'
}

with open('../training-image-segmentation/images/merged.png', "rb") as image_file:
    init_img = base64.b64encode(image_file.read())

with open('../training-image-segmentation/images/mask.png','rb') as image_file:
    mask_img = base64.b64encode(image_file.read())

prompt = "RAW photo, a photograph of a beach, ocean, sunset, highly detailed, close up shot."
negative_prompt = ''

make_request(prompt, negative_prompt, parameters, init_img=init_img, mask_img=mask_img)