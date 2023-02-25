from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler
import PIL
import requests
from io import BytesIO
import torch

pipe = DiffusionPipeline.from_pretrained("realistic_vision_v1.3_checkpoint", custom_pipeline="stable_diffusion_mega", torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
#pipe.enable_attention_slicing()
pipe.safety_checker = None
generator = torch.Generator("cuda").manual_seed(47)
prompt = "RAW photo, a close up portrait photo of brutal 45 y.o man in wastelander clothes, long haircut, pale skin, slim body, background is city ruins, high detailed skin, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
negative_prompt = "nude, naked, deformed, bad hands, bad fingers, monochrome:1.3, oversaturated:1.3, bad hands, lowers, 3d render, cartoon, long body, blurry, duplicate, duplicate body parts, disfigured, poorly drawn, extra limbs, fused fingers, extra fingers, twisted, malformed hands, mutated hands and fingers, contorted, conjoined, missing limbs, logo, signature, text, words, low res, boring, mutated, artifacts, bad art, gross, ugly, poor quality, low quality, child"
images = pipe.text2img(prompt=prompt,negative_prompt=negative_prompt,guidance_scale=7.0, num_inference_steps=25, generator=generator).images
images[0].save("realistic.png")
