from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("./model", safety_checker=None, from_flax=True, dtype=torch.float16).to("cuda")

prompt = "a photo of sks man wearing a suit and sunglasses, highly detailed, close up shot"
negative_prompt="rendered, unrealistic, work of art, artistic, cinematic"

image = pipe(prompt,negative_prompt=negative_prompt).images[0]

image.save("output.png")

#**** Optionally, you can save the torch weights and use them directly

# pipe = StableDiffusionPipeline.from_pretrained("./model", safety_checker=None, from_flax=True, dtype=torch.float16).to("cuda")

# Save torch weights
#pipe.save_pretrained("model_torch")

#pipe = StableDiffusionPipeline.from_pretrained("./model_torch", safety_checker=None, dtype=torch.float16).to("cuda")

# prompt = "a photo of sks man wearing a suit and sunglasses, highly detailed, close up shot"
# negative_prompt="rendered, unrealistic, work of art, artistic, cinematic"

# image = pipe(prompt,negative_prompt=negative_prompt).images[0]

# image.save("output.png")