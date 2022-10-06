import os
import uuid
import base64
import json

from io import BytesIO
import PIL
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat

import numpy as np
import torch
from torch import autocast
from pytorch_lightning import seed_everything

def load_img(base64_image_str):
    img_bytes = base64.b64decode(base64_image_str)
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    image = Image.open(img_file)   # img is now PIL Image object
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def img2img(model, sampler, image, prompt, config, outpath, uc=[""]):
    DEFAULT_DDIM_STEPS = 50
    DEFAULT_SEED = 42
    # eta=0.0 corresponds to deterministic sampling
    DEFAULT_DDIM_ETA = 0.0
    DEFAULT_N_ITER = 2
    DEFAULT_N_SAMPLES=3
    DEFAULT_SCALE=7.5

    DEFAULT_STRENGTH = .75

    seed = config.get('seed', DEFAULT_SEED)
    seed_everything(seed)

    n_samples = config.get('n_samples', DEFAULT_N_SAMPLES)
    batch_size = n_samples
    n_iter = config.get('n_iter', DEFAULT_N_ITER)

    ddim_steps = config.get('ddim_steps', DEFAULT_DDIM_STEPS)
    scale = config.get('scale', DEFAULT_SCALE)
    ddim_eta = config.get('ddim_eta', DEFAULT_DDIM_ETA)
    strength = config.get('strength',DEFAULT_STRENGTH)

    data = [batch_size * [prompt]]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = 0

    unique_id = str(uuid.uuid4())[:8]

    init_image = load_img(image).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            for _ in trange(n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * uc)
                    if isinstance(prompts, tuple):
                            prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                    #encode
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,)
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(),
                        'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(outpath,f"{unique_id}-{base_count:05}.png"))
                        base_count += 1
    retval = []
    for i in range(base_count-1,-1,-1):
        img_path = os.path.join(outpath,f"{unique_id}-{i:05}.png")
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read())
            retval.append(base64_image)
    
    return retval    
