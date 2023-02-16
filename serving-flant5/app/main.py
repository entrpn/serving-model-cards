import os
from fastapi import FastAPI, Request

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = os.getenv("MODEL_NAME")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,from_tf=False, from_flax=False).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name,from_tf=False, from_flax=False)

default_config = {
    "min_length" : 150,
    "max_length" : 200,
    "temperature" : 1,
    "top_k" : 15,
    "top_p" : 0.75,
    "repetition_penalty" : 7.0,
    "num_beams" : 25,
    "skip_special_tokens" : True
}

app = FastAPI()

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    """
    Request: 
    {
        "instances" : [
            {
                "prompt" : "Generate an intro paragraph about dinosaurs:", 
                "force_words" : ["lion", "aliens", "purple"],
                "parameters" : {
                    "min_length" : 150,
                    "max_length" : 200,
                    "temperature" : 1,
                    "top_k" : 15,
                    "top_p" : 0.75,
                    "repetition_penalty" : 7.0,
                    "num_beams" : 25,
                    "skip_special_tokens" : true
                }
            }
        ]
    }
    """
    body = await request.json()

    instances = body["instances"]

    retval = []

    for instance in instances:
        prompt = instance.get("prompt",None)
        config = instance.get("parameters")
        force_words = instance.get("force_words",[])

        if not config:
            config = {}

        max_length=config.get("max_length",default_config["max_length"])
        min_length=config.get("min_length",default_config["min_length"])
        temperature=config.get("temperature",default_config["temperature"])
        top_k=config.get("top_k",default_config["top_k"])
        top_p=config.get("top_p",default_config["top_p"])
        repetition_penalty=config.get("repetition_penalty",default_config["repetition_penalty"])
        num_beams=config.get("num_beams",default_config["num_beams"])
        skip_special_tokens = config.get("skip_special_tokens",default_config["skip_special_tokens"])
        

        inputs = tokenizer(prompt,return_tensors="pt").to(device)
        if force_words:
            force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                force_words_ids=force_words_ids,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams
            )
        else:
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams
            )

        output = tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)

        retval.append({"prompt" : prompt, "output" : output, "parameters" : config})
    return {"predictions" : retval}