import requests
import os
def make_request(prompts, parameters, force_words=None):
    instances = []
    for prompt in prompts:
        instance = {
            "prompt" : prompt,
            "force_words" : force_words,
        }
        instances.append(instance)
    app_dict = {
        "instances" : instances
    }
    app_dict["instances"][0]["parameters"] = parameters

    r = requests.post("http://127.0.0.1/predict", json=app_dict)
    print(r.status_code, r.reason)
    print(r.text)

parameters = {
    "min_length" : 150,
    "max_length" : 200,
    "temperature" : 1,
    "top_k" : 15,
    "top_p" : 0.75,
    "repetition_penalty" : 7.0,
    "num_beams" : 25,
    "skip_special_tokens" : True
}

prompt = "Generate an fictional story about a little girl and her dinosaur:"

make_request(prompts=[prompt,prompt], parameters=parameters)

