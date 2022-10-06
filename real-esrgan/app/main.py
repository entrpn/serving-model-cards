
from fastapi import FastAPI, Request

from inference_realesrgan import inference_realesrgan

import uuid
import base64
import json
import glob
import os
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

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
            {"image" : "base64encodedimage"}
        ],
        "parameters" : {
            "face_enhance" : true,
            "tile" : 0,
            "tile_pad" : 10,
            "prepad" : 0,
            "fp32" : true,
            "outscale" : 4
        }
    }
    """
    body = await request.json()

    instances = body["instances"]
    config =  body["parameters"]
    logger.debug(f"config : {config}")

    retval, error = inference_realesrgan(instances[0]['image'], config)
    
    if error:
        retval = ''
    else:
        error = ''

    return {"predictions" : retval, "error" : error}