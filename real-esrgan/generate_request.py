import base64
import json

with open('../images/ddlm_2.png', "rb") as image_file:
    base64_image = base64.b64encode(image_file.read())
print(type(base64_image))
appDict = {
  "instances": [{"image" : base64_image.decode("utf-8") }],
  "parameters": {
      "face-enhance" : True,
      "tile" : 0,
      "tile-pad" : 10,
      "prepad" : 0,
      "fp32" : True,
      "outscale" : 4
  },
}
app_json = json.dumps(appDict)
with open("request.json", "w") as text_file:
    text_file.write(app_json)