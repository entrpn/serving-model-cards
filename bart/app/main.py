import os
from fastapi import FastAPI, Request

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)

logging.info(f"AIP_PREDICT_ROUTE: {os.environ['AIP_PREDICT_ROUTE']}")

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    """
    Request: 
    {
        "instances" : [
            {"text" : "",
            }
        ,
            "parameters" : {
                "text_max_length" : 1024,
                "num_beams" : 2,
                "min_length" : 0,
                "max_length" : 3,
        }]
    }
    """
    body = await request.json()
    instances = body["instances"]
    retval = []
    for instance in instances:
        config = instance["parameters"]
        article = instance['text']
        text_max_length = config.get('text_max_length', 2014)
        num_beams = config.get('num_beams', 2)
        min_length = config.get('min_length', 10)
        max_length = config.get('max_length', 50)
        
        inputs = tokenizer(article, max_length=text_max_length, truncation=True, return_tensors="pt").to(device)
        summary_ids = model.generate(inputs['input_ids'], num_beams=num_beams, min_length=min_length, max_length=max_length)
        summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        retval.append({"summary" : summary, 'parameters' : config})

    return {"predictions" : retval}
