import json
import uuid
import os

from diffusers import StableDiffusionPipeline
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
import torch
from PIL import Image

from google.cloud import storage

HF_TOKEN = os.getenv('HF_TOKEN')
GCS_OUTPUT_DIR = os.getenv('GCS_OUTPUT_DIR')

if HF_TOKEN is None:
    raise Exception("No Huggingface token found, exiting...")

if GCS_OUTPUT_DIR is None:
    raise Exception("No GCS_OUTPUT_DIR found, exiting...")

bucket_name = GCS_OUTPUT_DIR.replace("gs://",'').split('/')[0]
subfolders = GCS_OUTPUT_DIR.replace("gs://",'').split('/')[1:]
subfolders = "/".join(subfolders)
if not subfolders.endswith(os.path.sep):
    subfolders += os.path.sep

print(bucket_name)
print(subfolders)

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

metadata_blob = bucket.blob(f"{subfolders}results.jsonl")
retval = []

def get_scheduler(model_id, scheduler_name):
    scheduler = None
    if scheduler_name == 'DPMSolverMultistepScheduler':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == 'EulerAncestralDiscreteScheduler':
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == 'EulerDiscreteScheduler':
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == 'LMSDiscreteScheduler':
        scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif scheduler_name == 'PNDMScheduler':
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    return scheduler

def infer(key, lines):
    model_id, scheduler_name = key
    scheduler = get_scheduler(model_id, scheduler_name)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=scheduler, use_auth_token=HF_TOKEN)
    pipe.safety_checker = None
    pipe = pipe.to("cuda")
    # don't recreate scheduler if same
    for line in lines:
        prompt = line['prompt']
        if prompt is None:
            continue
        
        negative_prompt = line.get('negative_prompt',None)
        num_images = line.get('num_images',4)
        scale = line.get('scale',7.5)
        seed = line.get('seed',None)
        num_inference_steps = line.get('num_inference_steps', 50)
        
        generator = None
        if seed:
            generator = torch.Generator("cuda").manual_seed(seed)
        pipe.scheduler = scheduler
        images = pipe(prompt, 
                     negative_prompt=negative_prompt, 
                     num_images_per_prompt=num_images, 
                     scale=scale, 
                     num_inference_steps=num_inference_steps,
                     generator=generator).images
        print(len(images))
        image_uris = []
        for image in images:
            uid = uuid.uuid4()
            local_filename = f"/tmp/{uid}.png" 
            image.save(local_filename)
            blob = bucket.blob(f"{subfolders}{uid}.png")
            blob.upload_from_filename(local_filename)
            image_uris.append(f"gs://{bucket_name}/{subfolders}{uid}.png")
        
        line['image_uris'] = image_uris
        retval.append(json.dumps(line))
    with metadata_blob.open("w") as f:
        f.write('\n'.join(retval))


lines_dict = {}
with open('metadata.jsonl','r') as f:
    for line in f:
        print(json.loads(line))
        line_dict = json.loads(line)
        model_id = line_dict["model_id"]
        scheduler_name = line_dict.get('scheduler','DDIMScheduler')
        if (model_id, scheduler_name) in lines_dict.keys():
            lines_dict[(model_id, scheduler_name)].append(line_dict)
        else:
            lines_dict[(model_id, scheduler_name)] = [line_dict]
    
for key in lines_dict.keys():
    infer(key, lines_dict[key])