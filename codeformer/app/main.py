
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

from face_restoration import face_restore

import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                    connect_list=['32', '64', '128', '256']).to(device)
ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                model_dir='weights/CodeFormer', progress=True, file_name=None)
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

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