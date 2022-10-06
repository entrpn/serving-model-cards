import base64
import json

with open('../images/01.jpeg', "rb") as image_file:
    img = base64.b64encode(image_file.read())

appDict = {
    "instances": [
        {
            "image" : img.decode('utf-8'),
            "parameters": {
                "w" : 0.7,
                "upscale" : 2,
                "has_aligned" : True,
                "only_center_face" : False,
                "draw_box" : False,
                "bg_upsampler" : "realesrgan",
                "face_upsample" : False,
                "bg_tile" : 400
            }
        }
    ],
}

app_json = json.dumps(appDict)
with open("request.json", "w") as text_file:
    text_file.write(app_json)