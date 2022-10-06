import os
import uuid
import base64
import json

from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange

import numpy as np
import torch
from torch import autocast
from pytorch_lightning import seed_everything


def txt2img(model, sampler, prompt, config, outpath, uc=[""]):
    DEFAULT_DDIM_STEPS = 50
    DEFAULT_SEED = 42
    # eta=0.0 corresponds to deterministic sampling
    DEFAULT_DDIM_ETA = 0.0
    DEFAULT_N_ITER = 2
    DEFAULT_H = 512
    DEFAULT_W = 512
    # latent channels
    DEFAULT_C = 4
    # Downsampling factor
    DEFAULT_F=8
    DEFAULT_N_SAMPLES=3
    DEFAULT_SCALE=7.5

    # TODO - still unsure how this works with multiple requests. Are they queued up, or is seed_everthing called while one request is doing inference? This is related that I'm using async def for the route.
    seed = config.get('seed', DEFAULT_SEED)
    seed_everything(seed)

    n_samples = config.get('n_samples', DEFAULT_N_SAMPLES)
    batch_size = n_samples
    n_iter = config.get('n_iter', DEFAULT_N_ITER)

    ddim_steps = config.get('ddim_steps', DEFAULT_DDIM_STEPS)
    scale = config.get('scale', DEFAULT_SCALE)
    ddim_eta = config.get('ddim_eta', DEFAULT_DDIM_ETA)

    C = config.get('C', DEFAULT_C)
    H = config.get('H', DEFAULT_H)
    W = config.get('W', DEFAULT_W)
    f = config.get('f', DEFAULT_F)

    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = 0

    unique_id = str(uuid.uuid4())[:8]

    start_code = None
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for _ in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = model.get_learned_conditioning(batch_size * uc)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        x_t=start_code
                                                        )
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(),
                            'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(outpath,f"{unique_id}-{base_count:05}.png"))
                            base_count += 1
    retval = []
    for i in range(base_count-1,-1,-1):
        img_path = os.path.join(outpath,f"{unique_id}-{i:05}.png")
        with open(img_path, "rb") as image_file:
            print("encoding image")
            base64_image = base64.b64encode(image_file.read())
            retval.append(base64_image)
    
    return retval

