import base64
import json

with open('../images/ddlm_2.png', "rb") as image_file:
    ddlm_image = base64.b64encode(image_file.read())

appDict = {
    "instances": [
        # {
        #     "prompt" : "A woman dressed like the Mexican Holiday Dia de los Muertos", 
        #     "image" : ddlm_image.decode("utf-8"),
        #     "parameters": {
        #         "ddim_steps" : 50,
        #         "scale" : 7.5,
        #         "n_samples" : 2,
        #         "n_itter" : 2,
        #         "strength" : .55,
        #         "type" : "img2img"
        #     }
        # },
        {
            "prompt" : "An apocalyptic alien city ravaged by time, highly detailed, cinematic, intricate, trending in artstation",
            "parameters": {
                "ddim_steps" : 30,
                "scale" : 7.5,
                "n_samples" : 2,
                "n_itter" : 2,
                "type" : "txt2img"
            }
        }
    ],
}

app_json = json.dumps(appDict)
with open("request.json", "w") as text_file:
    text_file.write(app_json)