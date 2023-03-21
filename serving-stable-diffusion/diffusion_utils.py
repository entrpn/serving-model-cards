from diffusers import DiffusionPipeline
import PIL
import requests
from io import BytesIO
import torch
import base64
from io import BytesIO

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

def get_scheduler(pipe, scheduler_name):
    scheduler = None
    if scheduler_name == 'DPMSolverMultistepScheduler':
        scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'EulerAncestralDiscreteScheduler':
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'EulerDiscreteScheduler':
        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'LMSDiscreteScheduler':
        scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'PNDMScheduler':
        scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    else:
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return scheduler

def base64_image_to_pil(base64_image_str):
    img_bytes = base64.b64decode(base64_image_str)
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    image = PIL.Image.open(img_file).convert("RGB")   # img is now PIL Image object
    return image

def pil_image_to_base64(images):
    retval = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        retval.append(img_str)
    return retval

def get_pipeline(model_name, use_cuda=True, revision="main", use_xformers=False):
    if revision == "fp16":
        torch_dtype=torch.float16
    elif revision == "bf16":
        torch_dtype=torch.bfloat16
    else:
        torch_dtype=torch.float32

    pipe = DiffusionPipeline.from_pretrained(model_name, custom_pipeline="stable_diffusion_mega", revision=revision, torch_dtype=torch_dtype)
    if use_cuda:
        pipe = pipe.to("cuda")
        if use_xformers:
            pipe.enable_xformers_memory_efficient_attention()

    return pipe

def image_to_image(
    pipe, 
    prompt, 
    negative_prompt,
    steps,
    scale,
    seed,
    num_images,
    width,
    height,
    scheduler_name,
    eta,
    init_image,
    strength
     ):

    if scheduler_name is not None:
        pipe.scheduler = get_scheduler(pipe, scheduler_name)
    
    generator = torch.Generator(seed)

    images = pipe.img2img(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
        eta=eta,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        image=base64_image_to_pil(init_image), 
        strength=strength).images
    return pil_image_to_base64(images)

def text_to_image(
    pipe, 
    prompt, 
    negative_prompt, 
    steps, 
    scale, 
    seed, 
    num_images, 
    width, 
    height, 
    scheduler_name, 
    eta):
    
    if scheduler_name is not None:
        pipe.scheduler = get_scheduler(pipe, scheduler_name)
    
    generator = torch.Generator(seed)

    images = pipe.text2img(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
        eta=eta,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images).images
    return pil_image_to_base64(images)

def inpaint(pipe, 
    prompt,
    negative_prompt, 
    steps, scale, 
    seed, 
    num_images, 
    scheduler_name, 
    eta,
    strength,
    init_image,
    mask_image):

    if scheduler_name is not None:
        pipe.scheduler = get_scheduler(pipe, scheduler_name)
    
    generator = torch.Generator(seed)

    images = pipe.inpaint(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
        eta=eta,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        image=base64_image_to_pil(init_image), mask_image=base64_image_to_pil(mask_image), strength=strength).images
    return pil_image_to_base64(images)