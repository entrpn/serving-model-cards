
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
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
import torch.nn as nn
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from txt2img import txt2img
from img2img import img2img

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt", device)

txt2img_outdir = "/outputs/txt2img-samples"
img2img_outdir = "/outputs/img2img-samples"

os.makedirs(txt2img_outdir, exist_ok=True)
os.makedirs(img2img_outdir, exist_ok=True)

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
            {"prompt" : "a dog wearing a dress", "image" : "base64encodedimage", "uc" : ["fog", "grainy", "purple"]}
        ],
        "parameters" : {
            "ddim_steps" : 50,
            "scale" : 7.5,
            "type" : "txt2img"
        }
    }
    """
    body = await request.json()

    instances = body["instances"]
    
    retval = []

    for instance in instances:
        prompt = instance["prompt"]
        uc = instance.get("uc",[""])
        config = instance["parameters"]
        infer_type = config.get('type',"txt2img")
        images = []
        if infer_type == 'txt2img':
            sampler = PLMSSampler(model)
            images = txt2img(model=model,
                            sampler=sampler,
                            prompt=prompt,
                            config=config,
                            outpath=txt2img_outdir,
                            uc=uc)

        elif infer_type == 'img2img':
            image = instances[0]["image"]
            sampler = DDIMSampler(model)
            images = img2img(model=model,
                            sampler=sampler,
                            image=image,
                            prompt=prompt,
                            config=config,
                            outpath=img2img_outdir,
                            uc=uc)
        retval.append({"prompt" : prompt, "images" : images})
    return {"predictions" : retval}