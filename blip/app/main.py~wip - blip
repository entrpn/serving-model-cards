
from fastapi import FastAPI, Request

import uuid
import base64
import json
import numpy as np
import pickle
import os
import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
import time

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        ,
            "parameters" : {
                "w" : 0.5,
                "upscale" : 2,
                "has_aligned" : false,
                "only_center_face" : false,
                "draw_box" : false,
                "bg_upsampler" "None",
                "face_upsample" : false,
                "bg_tile" : 400
        }]
    }
    """
    body = await request.json()

    instances = body["instances"]
    
    retval = []
    for instance in instances:
        image = instance["image"]
        config = instance["parameters"]

        image, error = face_restore(net, image, config)
        retval.append(image)
    return {"predictions" : retval, "error" : error}