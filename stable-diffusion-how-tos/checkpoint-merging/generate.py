from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import PIL
import requests
from io import BytesIO
import torch

pipe = DiffusionPipeline.from_pretrained("refined_checkpoint", custom_pipeline="stable_diffusion_mega", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
#pipe.enable_attention_slicing()
pipe.safety_checker = None
generator = torch.Generator("cuda").manual_seed(47)
prompt = "analog style, polaroid photo, close up shot, of enticing 1960s 25 year old female hippie, detailed skin, wearing vintage psychedelic micro long pants and shirt, barefoot, pretty face, slim physique, natural lighting, film grain, photographed on a SX-70 One-Step, 103mm lens, f/14.6 aperture"
negative_prompt = "nude, naked, deformed, bad hands, bad fingers, monochrome:1.3, oversaturated:1.3, bad hands, lowers, 3d render, cartoon, long body, blurry, duplicate, duplicate body parts, disfigured, poorly drawn, extra limbs, fused fingers, extra fingers, twisted, malformed hands, mutated hands and fingers, contorted, conjoined, missing limbs, logo, signature, text, words, low res, boring, mutated, artifacts, bad art, gross, ugly, poor quality, low quality, child"
images = pipe.text2img(prompt=prompt,negative_prompt=negative_prompt,guidance_scale=8.0, num_inference_steps=75, generator=generator).images
images[0].save("refined.png")
