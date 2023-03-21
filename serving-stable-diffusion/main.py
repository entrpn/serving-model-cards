
from fastapi import FastAPI, Request

import uuid
import base64
import json
import numpy as np
import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
import time

from diffusion_utils import get_pipeline, image_to_image, text_to_image, inpaint

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

MODEL_NAME = os.getenv("MODEL_NAME",None)
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
USE_XFORMERS = os.getenv("USE_XFORMERS",False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipe = get_pipeline(model_name=MODEL_NAME, revision=MODEL_REVISION, use_xformers=USE_XFORMERS)

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
            {"prompt" : "a dog wearing a dress", 
            "image" : "base64encodedimage", 
            "init_img" : "base64encodedimage", # img2img and inpainting
            "mask_img" : "base64encodedimg", # inpainting
            "negative_prompt" : ["fog grainy purple"],
            "parameters" : {
                "type" : "txt2img", #txt2img, img2img, inpaint
                "steps" : 50,
                "scale" : 7.5,
                "seed" : 47,
                "num_images" : 4,
                "eta" : 0.0,
                "width" : 512,
                "height" : 512,
                "strength" : 0.75 # img2img, inpaint,
                "scheduler" : "DPMSolverMultistepScheduler" # Supported DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
                }
            }
        ]
    }
    """
    body = await request.json()

    instances = body["instances"]
    
    retval = []

    for instance in instances:
        prompt = instance["prompt"]
        negative_prompt = instance.get("negative_prompt","")
        config = instance["parameters"]
        print("config:",config)
        infer_type = config.get('type',"txt2img")
        steps = config.get("steps",50)
        scale = config.get("scale",7.5)
        seed = config.get("seed",0)
        width = config.get("width", 512)
        height = config.get("height", 512)
        num_images = config.get("num_images",4)
        eta=config.get("eta",0.0)
        scheduler_name = config.get("scheduler","DDIMScheduler")
        images = []
        if infer_type == 'txt2img':
            images = text_to_image(
                pipe=pipe, 
                prompt=prompt, 
                negative_prompt=negative_prompt,
                steps=steps,
                scale=scale,
                seed=seed,
                num_images=num_images,
                width=width,
                height=height,
                scheduler_name=scheduler_name,
                eta=eta)
        elif infer_type == 'img2img':
            init_image =instance["init_img"]
            strength = config.get('strength',0.75)
            images = image_to_image(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                scale=scale,
                seed=seed,
                num_images=num_images,
                width=width,
                height=height,
                scheduler_name=scheduler_name,
                eta=eta,
                init_image=init_image,
                strength=strength
            )
        elif infer_type == 'inpainting':
            init_image =instance["init_img"]
            strength = config.get('strength',0.75)
            mask_image = instance["mask_img"]
            images = inpaint(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                scale=scale,
                seed=seed,
                num_images=num_images,
                scheduler_name=scheduler_name,
                eta=eta,
                strength=strength,
                init_image=init_image,
                mask_image=mask_image
            )
        retval.append({"instance" : instance, "images" : images})
    return {"predictions" : retval}