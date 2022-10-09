
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

from infer import caption, qna, img2txt_matching, feature_extraction

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
            {"image" : "base64encodedimage",
             "questions" : ["Where is the woman sitting?"],
             "captions" : ["a woman sitting on the beach with a dog"]
            }
        ,
            "parameters" : {
                "type" : "captioning|qna|img2txt_matching|feature_extraction",
                "sample" : false,
                "img_size" : 384,
                "num_beams" : 3,
                "max_length" : 20,
                "min_length" : 5
        }]
    }
    """
    body = await request.json()

    instances = body["instances"]
    
    retval = []
    for instance in instances:
        image = instance["image"]
        config = instance["parameters"]
        inference_type = config.get('type','captioning')
        if inference_type == 'captioning':
            image_size = config.get('img_size', 384)
            num_beams = config.get('num_beams', 3)
            max_length = config.get('max_length', 20)
            min_length = config.get('min_length', 5)
            ret = caption(image, image_size, num_beams, max_length, min_length)
        elif inference_type == 'qna':
            image_size = config.get('img_size', 480)
            question = instance['questions']
            ret = qna(image, image_size, question)
            ret['parameters'] = config
        elif inference_type == 'img2txt_matching':
            image_size = config.get('img_size', 384)
            captions = instance['captions']
            retval.append(img2txt_matching(image, image_size, captions))
        elif inference_type == 'feature_extraction':
            image_size = config.get('img_size', 224)
            captions = instance['captions']
            retval.append(feature_extraction(image, image_size, captions))
        ret['parameters'] = config
        retval.append(ret)

    return {"predictions" : retval}